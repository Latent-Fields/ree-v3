#!/opt/local/bin/python3
"""
V3-EXQ-325: SD-021 Descending Pain Modulation -- Commitment-Gated z_harm_s Attenuation

experiment_purpose: evidence

Tests that harm_descending_mod_enabled=True attenuates z_harm_s (sensory-discriminative)
during committed episodes, while z_harm_a (affective) is unaffected.

Two conditions per seed:
  DESCENDING -- harm_descending_mod_enabled=True, descending_attenuation_factor=0.5
  CONTROL    -- harm_descending_mod_enabled=False (current default)

Key metrics:
  z_harm_s_ratio -- mean z_harm_s_norm during committed / mean z_harm_s_norm during uncommitted
                    (DESCENDING should have lower ratio, i.e., z_harm_s suppressed during commit)
  z_harm_a_ratio -- same for z_harm_a (should NOT differ between conditions -- selectivity check)
  benefit_ratio  -- task performance (DESCENDING should not degrade or should improve)

Pass criterion (pre-registered):
  C1: z_harm_s_ratio_descending < z_harm_s_ratio_control (z_harm_s suppressed in committed)
  C2: |z_harm_a_ratio_descending - z_harm_a_ratio_control| < 0.3 (z_harm_a not attenuated)
  C3: n_committed_steps_descending >= 10 (agent does commit in DESCENDING condition)

Experiment PASS: >= 3/5 seeds satisfy C1, C2, and C3.

Note: Requires SD-022 limb_damage_enabled=True to avoid fully correlated harm streams.
Requires sufficient training to produce committed states.

Claims: SD-021 (descending modulation), MECH-090, SD-011
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


EXPERIMENT_TYPE = "v3_exq_325_sd021_descending_pain_modulation"
CLAIM_IDS = ["SD-021", "MECH-090", "SD-011"]

C1_threshold = 0.0    # DESCENDING z_harm_s_ratio < CONTROL z_harm_s_ratio
C2_threshold = 0.3    # z_harm_a_ratio difference < 0.3 (selectivity)
C3_threshold = 10     # at least 10 committed steps
PASS_MIN_SEEDS = 3

HARM_OBS_DIM = 51
HARM_OBS_A_DIM = 7    # SD-022 limb damage
Z_HARM_DIM = 32
Z_HARM_A_DIM = 16

SEEDS = [42, 43, 44, 45, 46]
TRAIN_EPISODES = 80   # need enough training to get committed states
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200
LR = 1e-3


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=10, num_hazards=4, num_resources=3,
        hazard_harm=0.1, resource_benefit=0.05,
        use_proxy_fields=True,
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
        seed=seed,
    )


def make_config(descending: bool) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=17,   # SD-022: body_obs_dim=17 when limb_damage_enabled
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
        harm_descending_mod_enabled=descending,
        descending_attenuation_factor=0.5,
    )


def run_training(agent: REEAgent, enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                 env: CausalGridWorldV2, device, n_eps: int):
    """Train agent to produce committed states (prediction error variance drops)."""
    prox_head = nn.Sequential(nn.Linear(Z_HARM_DIM, 1), nn.Sigmoid()).to(device)
    all_params = (
        list(agent.parameters())
        + list(enc_s.parameters())
        + list(enc_a.parameters())
        + list(prox_head.parameters())
    )
    opt = optim.Adam(all_params, lr=LR)

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            harm_obs = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")
            if harm_obs is None:
                break
            harm_obs = harm_obs.to(device)

            z_harm_s = enc_s(harm_obs.unsqueeze(0))
            z_harm_a = None
            if harm_obs_a is not None:
                harm_obs_a_t = harm_obs_a.to(device)
                res = enc_a(harm_obs_a_t.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res

            latent = agent.sense(obs_body, obs_world,
                                 obs_harm=harm_obs, obs_harm_a=harm_obs_a)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, _, done, _, obs_dict = env.step(action_idx)

            opt.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            harm_loss = F.mse_loss(prox_head(z_harm_s), harm_obs[-1:].unsqueeze(0))
            total = pred_loss + harm_loss
            if total.requires_grad:
                total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()

            if done:
                break


def eval_commitment_attenuation(agent: REEAgent, enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                                 env: CausalGridWorldV2, device, n_eps: int) -> Dict:
    """Measure z_harm_s and z_harm_a norms split by committed vs uncommitted state."""
    z_s_committed = []
    z_s_uncommitted = []
    z_a_committed = []
    z_a_uncommitted = []
    total_committed = 0

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            harm_obs = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")
            if harm_obs is None:
                break
            harm_obs = harm_obs.to(device)

            with torch.no_grad():
                z_harm_s = enc_s(harm_obs.unsqueeze(0))
                z_harm_a_val = None
                if harm_obs_a is not None:
                    harm_obs_a_t = harm_obs_a.to(device)
                    res = enc_a(harm_obs_a_t.unsqueeze(0))
                    z_harm_a_val = res[0] if isinstance(res, tuple) else res

                latent = agent.sense(obs_body, obs_world,
                                     obs_harm=harm_obs, obs_harm_a=harm_obs_a)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            is_committed = agent.e3._committed_trajectory is not None
            action_idx = int(action.argmax(dim=-1).item())
            _, _, done, _, obs_dict = env.step(action_idx)

            z_s_norm = float(z_harm_s.norm().item())
            z_a_norm = float(z_harm_a_val.norm().item()) if z_harm_a_val is not None else 0.0

            # Note: we measure the latent AFTER sense() which includes descending modulation
            # Use latent.z_harm for z_harm_s (already attenuated in DESCENDING condition)
            if latent.z_harm is not None:
                z_s_post = float(latent.z_harm.norm().item())
            else:
                z_s_post = z_s_norm
            if latent.z_harm_a is not None:
                z_a_post = float(latent.z_harm_a.norm().item())
            else:
                z_a_post = z_a_norm

            if is_committed:
                z_s_committed.append(z_s_post)
                z_a_committed.append(z_a_post)
                total_committed += 1
            else:
                z_s_uncommitted.append(z_s_post)
                z_a_uncommitted.append(z_a_post)

            if done:
                break

    def safe_ratio(committed, uncommitted):
        if not committed or not uncommitted:
            return 1.0
        mean_uc = float(np.mean(uncommitted))
        if mean_uc < 1e-8:
            return 1.0
        return float(np.mean(committed)) / mean_uc

    z_harm_s_ratio = safe_ratio(z_s_committed, z_s_uncommitted)
    z_harm_a_ratio = safe_ratio(z_a_committed, z_a_uncommitted)

    return {
        "z_harm_s_ratio": z_harm_s_ratio,
        "z_harm_a_ratio": z_harm_a_ratio,
        "n_committed_steps": total_committed,
        "z_harm_s_mean_committed": float(np.mean(z_s_committed)) if z_s_committed else 0.0,
        "z_harm_s_mean_uncommitted": float(np.mean(z_s_uncommitted)) if z_s_uncommitted else 0.0,
        "z_harm_a_mean_committed": float(np.mean(z_a_committed)) if z_a_committed else 0.0,
        "z_harm_a_mean_uncommitted": float(np.mean(z_a_uncommitted)) if z_a_uncommitted else 0.0,
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_train = 5 if dry_run else TRAIN_EPISODES
    n_eval = 3 if dry_run else EVAL_EPISODES

    print(f"Seed {seed}")
    condition_results = {}
    for condition in ["DESCENDING", "CONTROL"]:
        descending = (condition == "DESCENDING")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_env(seed)
        cfg = make_config(descending=descending)
        agent = REEAgent(cfg)
        enc_s = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM).to(device)
        enc_a = AffectiveHarmEncoder(harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM).to(device)

        print(f"  {condition}: training {n_train} eps...")
        run_training(agent, enc_s, enc_a, env, device, n_train)
        print(f"  {condition}: eval {n_eval} eps...")
        metrics = eval_commitment_attenuation(agent, enc_s, enc_a, env, device, n_eval)
        condition_results[condition] = metrics
        print(
            f"  {condition}: z_harm_s_ratio={metrics['z_harm_s_ratio']:.4f} "
            f"z_harm_a_ratio={metrics['z_harm_a_ratio']:.4f} "
            f"committed_steps={metrics['n_committed_steps']}"
        )

    desc = condition_results["DESCENDING"]
    ctrl = condition_results["CONTROL"]

    c1_pass = desc["z_harm_s_ratio"] < ctrl["z_harm_s_ratio"]
    c2_pass = abs(desc["z_harm_a_ratio"] - ctrl["z_harm_a_ratio"]) < C2_threshold
    c3_pass = desc["n_committed_steps"] >= C3_threshold
    seed_pass = c1_pass and c2_pass and c3_pass

    print(f"  -> {'PASS' if seed_pass else 'FAIL'}")
    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "z_harm_s_ratio_descending": desc["z_harm_s_ratio"],
        "z_harm_s_ratio_control": ctrl["z_harm_s_ratio"],
        "z_harm_a_ratio_descending": desc["z_harm_a_ratio"],
        "z_harm_a_ratio_control": ctrl["z_harm_a_ratio"],
        "n_committed_descending": desc["n_committed_steps"],
        "n_committed_control": ctrl["n_committed_steps"],
        "c1_z_harm_s_attenuated_in_committed": c1_pass,
        "c2_z_harm_a_selective": c2_pass,
        "c3_commits_exist": c3_pass,
        "condition_results": condition_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        "v3_exq_325_sd021_descending_pain_modulation_dry" if args.dry_run
        else f"v3_exq_325_sd021_descending_pain_modulation_{timestamp}_v3"
    )
    print(f"EXQ-325 start: {run_id}")

    per_seed = [run_seed(s, dry_run=args.dry_run) for s in SEEDS]
    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-325 {outcome} ===")
    print(f"Seeds pass: {seeds_passing}/{len(SEEDS)}")
    for r in per_seed:
        s = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {s} "
            f"s_ratio_D={r['z_harm_s_ratio_descending']:.4f} "
            f"s_ratio_C={r['z_harm_s_ratio_control']:.4f} "
            f"a_ratio_D={r['z_harm_a_ratio_descending']:.4f} "
            f"commits={r['n_committed_descending']}"
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
            "SD-021": evidence_direction,
            "MECH-090": evidence_direction,
            "SD-011": evidence_direction,
        },
        "outcome": outcome,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "registered_thresholds": {
            "C1_z_harm_s_ratio_reduced": C1_threshold,
            "C2_z_harm_a_selectivity": C2_threshold,
            "C3_committed_steps": C3_threshold,
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
