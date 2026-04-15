#!/opt/local/bin/python3
"""
V3-EXQ-323a: SD-019 Harm Stream Non-Redundancy Constraint -- SD-022 Substrate

experiment_purpose: evidence

Corrected version of EXQ-323. Root cause of EXQ-323 failure: AffectiveHarmEncoder
received 50-dim EMA proximity (structurally identical to harm_obs_s), so the
nonredundancy penalty had no discriminative signal to push against. r2_s_to_a was
structurally near 1.0 regardless of penalty (confirmed EXQ-241b, EXQ-241).

Fix: SD-022 substrate (limb_damage_enabled=True). harm_obs_a is now sourced from
4-directional body damage state (7 dims: damage[4]+max+mean+residual_pain), which
is causally independent of current world proximity. z_harm_a can encode a signal
that z_harm_s cannot, giving the nonredundancy penalty something to enforce.

SD-022 causal independence confirmed: EXQ-319 PASS (z_harm_a_retention_high=0.910
vs z_harm_a_retention_fresh=0.486 after hazard offset; dissociation_score > 0).

This experiment adds two fixes not in EXQ-323:
  1. limb_damage_enabled=True, heal_rate=0.001 (SD-022 substrate with slow healing)
  2. enc_a trained with explicit damage supervision (MSE on mean_damage scalar)
     without this, enc_a stays near random init (cos_sq ~ 0 by geometry, not independence)

Two conditions per seed:
  NONREDUNDANT -- harm_nonredundancy_weight=1.0 (cosine^2 penalty applied)
  BASELINE     -- harm_nonredundancy_weight=0.0 (no penalty)

Key metric:
  cosine_sq -- mean cosine_similarity^2 between z_harm_s and z_harm_a after training
               (lower is better -- streams are more orthogonal)

Pass criterion (pre-registered):
  C1: cosine_sq_nonredundant < cosine_sq_baseline  (penalty reduces redundancy)
  C2: cosine_sq_nonredundant <= 0.4                (absolute threshold)
  C3: cosine_sq_baseline > 0.05                    (baseline encoders genuinely correlated)
  C4: Both streams still have non-zero norms        (no collapse)

Experiment PASS: >= 3/5 seeds satisfy C1, C2, C3, and C4.

Claims: SD-019 (affective harm non-redundancy constraint), SD-011, SD-022
Supersedes: V3-EXQ-323 (old substrate, enc_a untrained -- invalid test)
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


EXPERIMENT_TYPE = "v3_exq_323a_sd019_harm_nonredundancy"
CLAIM_IDS = ["SD-019", "SD-011", "SD-022"]

# Pre-registered thresholds
C1_threshold = 0.0    # nonredundant cosine_sq must be < baseline
C2_threshold = 0.4    # nonredundant cosine_sq must be <= this
C3_threshold = 0.05   # baseline cosine_sq must be > this (encoders genuinely correlated)
C3_min_norm  = 0.01   # both streams must have non-zero norms (no collapse)
PASS_MIN_SEEDS = 3

# SD-022 substrate dims
HARM_OBS_DIM   = 51   # hazard_field(25) + resource_field(25) + harm_exposure(1)
HARM_OBS_A_DIM = 7    # SD-022: damage[4] + max_damage + mean_damage + residual_pain
MEAN_DAMAGE_IDX = 5   # index of mean_damage in harm_obs_a (damage[0..3], max, mean, residual)
Z_HARM_DIM   = 32
Z_HARM_A_DIM = 16
BODY_OBS_DIM = 17     # SD-022 extended: proxy(12) + damage[4] + residual_pain(1)

SEEDS = [42, 43, 44, 45, 46]
WARMUP_EPISODES   = 120   # more than EXQ-323's 60 -- SD-022 needs time to build damage signal
PRE_DAMAGE_TRANSITS = 20  # forced hazard transits after warmup to ensure non-zero damage
EVAL_STEPS        = 300
STEPS_PER_EPISODE = 200
LR = 1e-3


def make_env(seed: int) -> CausalGridWorldV2:
    """SD-022 substrate: limb_damage_enabled=True, slow healing, more hazards."""
    return CausalGridWorldV2(
        size=10,
        num_hazards=6,          # more contacts per episode vs EXQ-323's 4
        num_resources=3,
        hazard_harm=0.1,
        resource_benefit=0.05,
        use_proxy_fields=True,
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.001,        # slow healing -- SD-022 prediction: r2_s_to_a < 0.4
        seed=seed,
    )


def make_config(nonredundancy_weight: float) -> REEConfig:
    """REEConfig matching SD-022 substrate dims."""
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=250,
        action_dim=4,
        alpha_world=0.9,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
        use_affective_harm_stream=True,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        z_harm_a_dim=Z_HARM_A_DIM,
        limb_damage_enabled=True,
        harm_nonredundancy_weight=nonredundancy_weight,
    )


def _find_hazard_cell(env: CausalGridWorldV2):
    for hx, hy in env.hazards:
        return (int(hx), int(hy))
    return None


def _teleport_agent(env: CausalGridWorldV2, tx: int, ty: int) -> None:
    if env.contamination_grid[env.agent_x, env.agent_y] >= env.contamination_threshold:
        env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["contaminated"]
    else:
        env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["empty"]
    env.agent_x = tx
    env.agent_y = ty
    env.grid[tx, ty] = env.ENTITY_TYPES["agent"]


def run_pre_damage(env: CausalGridWorldV2, n_transits: int, dry_run: bool) -> float:
    """Force agent through hazard transits to seed non-zero limb damage state."""
    n = 3 if dry_run else n_transits
    hcell = _find_hazard_cell(env)
    if hcell is None:
        return 0.0
    hx, hy = hcell
    for _ in range(n):
        _teleport_agent(env, hx, hy)
        env.step(4)   # stay on hazard (accumulate damage)
        env.step(0)   # step away
        env.step(1)   # step back
        env.step(4)
    return float(np.mean(env.limb_damage))


def run_training(
    enc_s: HarmEncoder,
    enc_a: AffectiveHarmEncoder,
    harm_head: nn.Module,
    damage_head: nn.Module,
    env: CausalGridWorldV2,
    device,
    n_eps: int,
    nonredundancy_weight: float,
) -> None:
    """
    Train both encoders with supervised losses, optionally with nonredundancy penalty.

    enc_s: harm proximity supervision (MSE on harm_obs[-1])
    enc_a: damage supervision (MSE on mean_damage from harm_obs_a[MEAN_DAMAGE_IDX])
    penalty: cosine^2 between proj(z_harm_s) and proj(z_harm_a) when weight > 0

    Both encoders receive meaningful gradients so that baseline cosine_sq reflects
    genuine correlation between the learned representations (not random geometry).
    """
    all_enc_params = (
        list(enc_s.parameters())
        + list(enc_a.parameters())
        + list(harm_head.parameters())
        + list(damage_head.parameters())
    )
    opt = optim.Adam(all_enc_params, lr=LR)

    # Projection layers for nonredundancy loss (shared comparison dim = Z_HARM_DIM)
    proj_s = nn.Linear(Z_HARM_DIM, Z_HARM_DIM).to(device)
    proj_a = nn.Linear(Z_HARM_A_DIM, Z_HARM_DIM).to(device)
    opt_proj = optim.Adam(list(proj_s.parameters()) + list(proj_a.parameters()), lr=LR)

    _, obs_dict = env.reset()
    for ep in range(n_eps):
        _, obs_dict = env.reset()
        for _ in range(STEPS_PER_EPISODE):
            harm_obs   = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")
            if harm_obs is None:
                break
            harm_obs   = harm_obs.to(device)
            harm_obs_a = harm_obs_a.to(device) if harm_obs_a is not None else None

            z_harm_s = enc_s(harm_obs.unsqueeze(0))
            z_harm_a = None
            if harm_obs_a is not None:
                res = enc_a(harm_obs_a.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res

            # enc_s loss: predict harm proximity from z_harm_s
            prox_pred   = harm_head(z_harm_s)
            prox_target = harm_obs[-1:].unsqueeze(0)
            loss_s = F.mse_loss(prox_pred, prox_target)

            # enc_a loss: predict mean_damage from z_harm_a (body damage signal)
            loss_a = torch.tensor(0.0, device=device)
            if z_harm_a is not None and harm_obs_a is not None:
                dmg_pred   = damage_head(z_harm_a)
                dmg_target = harm_obs_a[MEAN_DAMAGE_IDX:MEAN_DAMAGE_IDX + 1].unsqueeze(0)
                loss_a = F.mse_loss(dmg_pred, dmg_target)

            total_loss = loss_s + loss_a

            # Nonredundancy penalty (SD-019): cosine^2 between projected streams
            if nonredundancy_weight > 0.0 and z_harm_a is not None:
                z_s_proj = proj_s(z_harm_s)
                z_a_proj = proj_a(z_harm_a)
                cos_sim  = F.cosine_similarity(z_s_proj, z_a_proj, dim=-1).mean()
                penalty  = cos_sim.pow(2)
                total_loss = total_loss + nonredundancy_weight * penalty

            opt.zero_grad()
            opt_proj.zero_grad()
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_enc_params, 1.0)
                opt.step()
                if nonredundancy_weight > 0.0:
                    opt_proj.step()

            action_idx = random.randint(0, 3)
            _, _, done, _, obs_dict = env.step(action_idx)
            if done:
                break


def eval_redundancy(
    enc_s: HarmEncoder,
    enc_a: AffectiveHarmEncoder,
    env: CausalGridWorldV2,
    device,
) -> Dict:
    """Measure cosine_sq between z_harm_s and z_harm_a over EVAL_STEPS."""
    cosine_sqs = []
    z_s_norms  = []
    z_a_norms  = []
    harm_obs_a_norms = []   # diagnostic: is enc_a getting real input?

    _, obs_dict = env.reset()
    for _ in range(EVAL_STEPS):
        harm_obs   = obs_dict.get("harm_obs")
        harm_obs_a = obs_dict.get("harm_obs_a")
        if harm_obs is None:
            break
        harm_obs   = harm_obs.to(device)
        harm_obs_a = harm_obs_a.to(device) if harm_obs_a is not None else None

        with torch.no_grad():
            z_harm_s = enc_s(harm_obs.unsqueeze(0))
            z_harm_a = None
            if harm_obs_a is not None:
                res = enc_a(harm_obs_a.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res
                harm_obs_a_norms.append(float(harm_obs_a.norm().item()))

            if z_harm_a is not None:
                min_d = min(z_harm_s.shape[-1], z_harm_a.shape[-1])
                cos_sq = float(
                    F.cosine_similarity(
                        z_harm_s[..., :min_d], z_harm_a[..., :min_d], dim=-1
                    ).pow(2).mean().item()
                )
                cosine_sqs.append(cos_sq)

            z_s_norms.append(float(z_harm_s.norm().item()))
            if z_harm_a is not None:
                z_a_norms.append(float(z_harm_a.norm().item()))

        action_idx = random.randint(0, 3)
        _, _, done, _, obs_dict = env.step(action_idx)
        if done:
            break

    return {
        "cosine_sq":        float(np.mean(cosine_sqs))        if cosine_sqs        else 1.0,
        "z_harm_s_norm":    float(np.mean(z_s_norms))         if z_s_norms         else 0.0,
        "z_harm_a_norm":    float(np.mean(z_a_norms))         if z_a_norms         else 0.0,
        "harm_obs_a_norm":  float(np.mean(harm_obs_a_norms))  if harm_obs_a_norms  else 0.0,
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_train = 5 if dry_run else WARMUP_EPISODES

    print(f"Seed {seed}", flush=True)

    condition_results = {}
    for condition in ["NONREDUNDANT", "BASELINE"]:
        weight = 1.0 if condition == "NONREDUNDANT" else 0.0
        # Re-seed both conditions identically so they start from same init
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = make_env(seed)
        # REEConfig needed only to confirm dims match; encoders are standalone
        cfg = make_config(nonredundancy_weight=weight)

        enc_s     = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM).to(device)
        enc_a     = AffectiveHarmEncoder(
            harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM
        ).to(device)
        harm_head   = nn.Sequential(nn.Linear(Z_HARM_DIM, 1), nn.Sigmoid()).to(device)
        damage_head = nn.Sequential(nn.Linear(Z_HARM_A_DIM, 1), nn.Sigmoid()).to(device)

        print(f"  {condition}: training {n_train} eps (weight={weight})...", flush=True)
        run_training(enc_s, enc_a, harm_head, damage_head, env, device, n_train, weight)

        # Pre-damage: force some hazard transits so limb_damage is non-zero at eval
        dmg_mean = run_pre_damage(env, PRE_DAMAGE_TRANSITS, dry_run)
        print(f"  {condition}: pre-damage done, mean_limb_damage={dmg_mean:.4f}", flush=True)

        print(f"  {condition}: eval...", flush=True)
        metrics = eval_redundancy(enc_s, enc_a, env, device)
        condition_results[condition] = metrics
        print(
            f"  {condition}: cosine_sq={metrics['cosine_sq']:.4f} "
            f"z_s_norm={metrics['z_harm_s_norm']:.4f} "
            f"z_a_norm={metrics['z_harm_a_norm']:.4f} "
            f"harm_obs_a_norm={metrics['harm_obs_a_norm']:.4f}",
            flush=True,
        )

    nr = condition_results["NONREDUNDANT"]
    bl = condition_results["BASELINE"]

    c1_pass = nr["cosine_sq"] < bl["cosine_sq"]
    c2_pass = nr["cosine_sq"] <= C2_threshold
    c3_pass = bl["cosine_sq"] > C3_threshold
    c4_pass = (
        nr["z_harm_s_norm"] >= C3_min_norm
        and nr["z_harm_a_norm"] >= C3_min_norm
    )
    seed_pass = c1_pass and c2_pass and c3_pass and c4_pass

    print(
        f"  -> {'PASS' if seed_pass else 'FAIL'} "
        f"C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass}",
        flush=True,
    )
    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "cosine_sq_nonredundant": nr["cosine_sq"],
        "cosine_sq_baseline":     bl["cosine_sq"],
        "z_harm_s_norm_nr":       nr["z_harm_s_norm"],
        "z_harm_a_norm_nr":       nr["z_harm_a_norm"],
        "harm_obs_a_norm_nr":     nr["harm_obs_a_norm"],
        "harm_obs_a_norm_bl":     bl["harm_obs_a_norm"],
        "c1_cosine_sq_reduced":   c1_pass,
        "c2_abs_threshold":       c2_pass,
        "c3_baseline_correlated": c3_pass,
        "c4_no_collapse":         c4_pass,
        "condition_results":      condition_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        "v3_exq_323a_sd019_harm_nonredundancy_dry" if args.dry_run
        else f"v3_exq_323a_sd019_harm_nonredundancy_{timestamp}_v3"
    )
    print(f"EXQ-323a start: {run_id}", flush=True)

    per_seed = [run_seed(s, dry_run=args.dry_run) for s in SEEDS]
    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-323a {outcome} ===")
    print(f"Seeds pass: {seeds_passing}/{len(SEEDS)} (need {PASS_MIN_SEEDS})")
    for r in per_seed:
        s = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {s} "
            f"cos_sq_NR={r['cosine_sq_nonredundant']:.4f} "
            f"cos_sq_BL={r['cosine_sq_baseline']:.4f} "
            f"obs_a_norm={r['harm_obs_a_norm_nr']:.4f}"
        )

    evidence_direction = "supports" if experiment_passes else "does_not_support"
    output = {
        "run_id":               run_id,
        "experiment_type":      EXPERIMENT_TYPE,
        "architecture_epoch":   "ree_hybrid_guardrails_v1",
        "claim_ids":            CLAIM_IDS,
        "experiment_purpose":   "evidence",
        "supersedes":           "v3_exq_323_sd019_harm_nonredundancy",
        "evidence_direction":   evidence_direction,
        "evidence_direction_per_claim": {
            "SD-019": evidence_direction,
            "SD-011": evidence_direction,
            "SD-022": evidence_direction,
        },
        "outcome":              outcome,
        "timestamp_utc":        datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        ),
        "registered_thresholds": {
            "C1_cosine_sq_reduced":   C1_threshold,
            "C2_abs_cosine_sq":       C2_threshold,
            "C3_baseline_correlated": C3_threshold,
            "C4_min_norm":            C3_min_norm,
            "seeds_needed":           PASS_MIN_SEEDS,
        },
        "config": {
            "seeds":              SEEDS,
            "warmup_episodes":    WARMUP_EPISODES,
            "steps_per_episode":  STEPS_PER_EPISODE,
            "pre_damage_transits": PRE_DAMAGE_TRANSITS,
            "eval_steps":         EVAL_STEPS,
            "num_hazards":        6,
            "heal_rate":          0.001,
            "limb_damage_enabled": True,
            "harm_obs_a_dim":     HARM_OBS_A_DIM,
            "nonredundancy_weight_nr": 1.0,
            "enc_a_loss":         "MSE on harm_obs_a[5] (mean_damage)",
        },
        "per_seed_results":     per_seed,
        "seeds_passing":        seeds_passing,
        "experiment_passes":    experiment_passes,
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
