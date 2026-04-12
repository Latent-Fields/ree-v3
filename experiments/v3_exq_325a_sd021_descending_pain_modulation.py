#!/opt/local/bin/python3
"""
V3-EXQ-325a: SD-021 Descending Pain Modulation -- Commitment-Gated z_harm_s Attenuation
             (E2 world-forward training fix)

experiment_purpose: evidence

Supersedes: V3-EXQ-325
Root cause of EXQ-325 failure: same training loop bug as EXQ-321. Only
compute_prediction_loss() (E1 loss) was called. E2 world_forward was never
trained, update_running_variance() was never called. running_variance stayed
frozen at 0.50, commitment_threshold=0.40 was never reached, beta_gate was
never elevated, and the SD-021 descending modulation path
(sense() attenuates z_harm_s when beta_gate.is_elevated) was never exercised.
DESCENDING=CONTROL in all runs.

Fix: added E2 world-forward training loop (wf_buf + wf_optimizer + update_running_variance
per step); extended TRAIN_EPISODES from 80 to 300; ensured beta_gate_bistable=True
and limb_damage_enabled=True are both set (both needed to reach committed state with
elevated gate).

Tests that harm_descending_mod_enabled=True attenuates z_harm_s (sensory-discriminative)
during committed episodes, while z_harm_a (affective) is unaffected.

Two conditions per seed:
  DESCENDING -- harm_descending_mod_enabled=True, descending_attenuation_factor=0.5
  CONTROL    -- harm_descending_mod_enabled=False

Key metrics:
  z_harm_s_ratio -- mean z_harm_s_norm during committed / mean z_harm_s_norm during uncommitted
                    (DESCENDING should have lower ratio: z_harm_s suppressed during commit)
  z_harm_a_ratio -- same for z_harm_a (should NOT differ between conditions -- selectivity check)

Pass criterion (pre-registered):
  C1: z_harm_s_ratio_descending < z_harm_s_ratio_control (z_harm_s suppressed in committed)
  C2: |z_harm_a_ratio_descending - z_harm_a_ratio_control| < 0.3 (z_harm_a not attenuated)
  C3: n_committed_steps_descending >= 10 (agent does commit in DESCENDING condition)

Experiment PASS: >= 3/5 seeds satisfy C1, C2, and C3.

Mechanism under test: SD-021 (harm_stream.descending_modulation).
If this PASSes: supports SD-021 -- when beta_gate is elevated (MECH-090 gate fires),
z_harm_s is attenuated by descending_attenuation_factor, and this attenuation is
stream-selective (z_harm_a unchanged). MECH-090 is NOT separately tagged because
this experiment presupposes MECH-090 (it uses beta_gate.is_elevated as a known
committed-state signal, not as the variable being tested). SD-011 is NOT tagged
because SD-011 tests the existence of the dual streams, not their modulation pattern.

Claims: SD-021 only.
"""

import json
import sys
import random
import datetime
import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


EXPERIMENT_TYPE = "v3_exq_325a_sd021_descending_pain_modulation"
CLAIM_IDS = ["SD-021"]

C1_threshold = 0.0    # DESCENDING z_harm_s_ratio < CONTROL z_harm_s_ratio
C2_threshold = 0.3    # z_harm_a_ratio difference < 0.3 (selectivity)
C3_threshold = 10     # at least 10 committed steps
PASS_MIN_SEEDS = 3

HARM_OBS_DIM = 51
HARM_OBS_A_DIM = 7    # SD-022 limb damage
Z_HARM_DIM = 32
Z_HARM_A_DIM = 16

SEEDS = [42, 43, 44, 45, 46]
TRAIN_EPISODES = 300   # extended from 80 -- needed for rv to drop from 0.50 to < 0.40
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200
LR = 1e-3
WF_LR = 1e-3


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
    from ree_core.utils.config import HeartbeatConfig
    # beta_gate_bistable=True required: gate must elevate and HOLD so SD-021
    # attenuation path is reached during committed steps
    hb = HeartbeatConfig(beta_gate_bistable=True)
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
        heartbeat=hb,
    )


def run_training(agent: REEAgent, enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                 env: CausalGridWorldV2, device, n_eps: int,
                 n_steps: int = STEPS_PER_EPISODE):
    """Train agent with E2 world-forward loop so running_variance can drop below commit_threshold."""
    prox_head = nn.Sequential(nn.Linear(Z_HARM_DIM, 1), nn.Sigmoid()).to(device)
    all_params = (
        list(agent.parameters())
        + list(enc_s.parameters())
        + list(enc_a.parameters())
        + list(prox_head.parameters())
    )
    opt = optim.Adam(all_params, lr=LR)

    wf_opt = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=WF_LR,
    )

    # wf_buf: (z_world_prev, action_onehot, z_world_curr) tuples
    wf_buf: deque = deque(maxlen=2000)

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(n_steps):
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

            z_world_curr = latent.z_world.detach()

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((
                    z_world_prev.cpu(),
                    action_prev.cpu(),
                    z_world_curr.cpu(),
                ))

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, _, done, _, obs_dict = env.step(action_idx)

            # E1 + harm encoder losses
            opt.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            harm_loss = F.mse_loss(prox_head(z_harm_s), harm_obs[-1:].unsqueeze(0))
            total = pred_loss + harm_loss
            if total.requires_grad:
                total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()

            # E2 world-forward training -- enables running_variance to drop
            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_opt.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_opt.step()
                # Direct variance update -- breaks chicken-and-egg deadlock
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            z_world_prev = z_world_curr
            action_prev = action.detach()

            if done:
                break


def eval_commitment_attenuation(agent: REEAgent, enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                                env: CausalGridWorldV2, device, n_eps: int,
                                n_steps: int = STEPS_PER_EPISODE) -> Dict:
    """Measure z_harm_s and z_harm_a norms split by committed vs uncommitted state."""
    z_s_committed: List[float] = []
    z_s_uncommitted: List[float] = []
    z_a_committed: List[float] = []
    z_a_uncommitted: List[float] = []
    total_committed = 0

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(n_steps):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            harm_obs = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")
            if harm_obs is None:
                break
            harm_obs = harm_obs.to(device)

            with torch.no_grad():
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

            # Measure z_harm after sense() -- includes descending modulation when active
            if latent.z_harm is not None:
                z_s_post = float(latent.z_harm.norm().item())
            else:
                z_s_post = 0.0
            if latent.z_harm_a is not None:
                z_a_post = float(latent.z_harm_a.norm().item())
            elif z_harm_a_val is not None:
                z_a_post = float(z_harm_a_val.norm().item())
            else:
                z_a_post = 0.0

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
    n_train = 2 if dry_run else TRAIN_EPISODES
    n_eval = 1 if dry_run else EVAL_EPISODES
    n_steps = 5 if dry_run else STEPS_PER_EPISODE

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
        run_training(agent, enc_s, enc_a, env, device, n_train, n_steps=n_steps)
        rv = agent.e3._running_variance
        print(f"  {condition}: rv_after_train={rv:.4f} committed_flag={rv < agent.e3.commit_threshold}")
        print(f"  {condition}: eval {n_eval} eps...")
        metrics = eval_commitment_attenuation(agent, enc_s, enc_a, env, device, n_eval, n_steps=n_steps)
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
        "v3_exq_325a_sd021_descending_pain_modulation_dry" if args.dry_run
        else f"v3_exq_325a_sd021_descending_pain_modulation_{timestamp}_v3"
    )
    print(f"EXQ-325a start: {run_id}")

    seeds_to_run = SEEDS[:1] if args.dry_run else SEEDS
    per_seed = [run_seed(s, dry_run=args.dry_run) for s in seeds_to_run]
    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-325a {outcome} ===")
    print(f"Seeds pass: {seeds_passing}/{len(seeds_to_run)}")
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
        "supersedes": "v3_exq_325_sd021_descending_pain_modulation",
        "registered_thresholds": {
            "C1_z_harm_s_ratio_reduced": C1_threshold,
            "C2_z_harm_a_selectivity": C2_threshold,
            "C3_committed_steps": C3_threshold,
            "seeds_needed": PASS_MIN_SEEDS,
        },
        "outcome": outcome,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "per_seed_results": per_seed,
        "seeds_passing": seeds_passing,
        "experiment_passes": experiment_passes,
        "fix_notes": (
            "EXQ-325 had frozen running_variance=0.50 (no E2 world-forward training). "
            "Fix: added wf_buf + wf_opt + update_running_variance per step; "
            "extended TRAIN_EPISODES 80->300; added beta_gate_bistable=True to "
            "both conditions (needed for gate to hold once elevated). "
            "claim_ids reduced to [SD-021] only -- MECH-090 and SD-011 are "
            "prerequisites used as substrate, not the variable being tested."
        ),
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
