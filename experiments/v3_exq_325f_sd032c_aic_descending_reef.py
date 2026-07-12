#!/opt/local/bin/python3
"""
V3-EXQ-325f: SD-032c AIC-Analog -- Reef-Enriched Substrate.

experiment_purpose: evidence

Supersedes: V3-EXQ-325d (non_contributory due to monostrategy: agent adopts single
fixed route, preventing behavioral diversity measurement for C2/C3 descending
modulation criteria).

Reef enrichment (SD-054): reef_enabled=True adds coral-reef safe zones + food-attracted
hazard drift. Creates two behavioral attractors ("flee to reef" vs "forage") breaking
the monostrategy on 10x10 grid. reef_patch_radius=2 appropriate for 10x10.

Also fixes hardcoded world_obs_dim=250 in make_config(): now reads env.world_obs_dim
dynamically so reef's +25 dim increase propagates correctly into REEAgent construction.

Four conditions per seed (all use beta_gate_bistable=True + limb_damage):
  AIC_ON            -- use_aic_analog=True, harm_descending_mod_enabled=True,
                        aic_drive_protect_weight=1.0 (default, drive-dependence ON)
  AIC_DRIVE_ABLATED -- use_aic_analog=True, harm_descending_mod_enabled=True,
                        aic_drive_protect_weight=0.0 (drive-independent ablation)
  LEGACY            -- use_aic_analog=False, harm_descending_mod_enabled=True
                        (legacy SD-021 raw-beta-gate path, as in EXQ-325a)
  CONTROL           -- harm_descending_mod_enabled=False (no attenuation)

Key metrics:
  z_harm_s_ratio -- mean z_harm_s_norm during committed / during uncommitted
  z_harm_a_ratio -- same for z_harm_a (should NOT differ between conditions)
  aic_salience_mean -- mean aic_salience across episode (AIC conditions only)
  n_committed_steps -- commit substrate sanity

Pass criteria (pre-registered, unchanged from 325c):
  C1 (attenuation present): AIC_ON z_harm_s_ratio < CONTROL z_harm_s_ratio
  C2 (stream selectivity):  |AIC_ON z_harm_a_ratio - CONTROL z_harm_a_ratio| < 0.3
  C3 (commits exist):       AIC_ON n_committed_steps >= 10
  C4 (DRIVE-DEPENDENCE, falsification signature):
         |AIC_ON z_harm_s_ratio - AIC_DRIVE_ABLATED z_harm_s_ratio| > 0.02
         If C4 fails, the structural drive-dependence is absent.
  C5 (module non-trivial):  AIC_ON aic_salience_mean > 0.0

Experiment PASS: >= 2/3 seeds satisfy C1, C2, C3, C4, C5.

Claims: SD-032c (primary), SD-021 (subsumed; retest of EXQ-325a hypothesis).
"""

import json
import sys
import random
import datetime
import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_325f_sd032c_aic_descending_reef"
CLAIM_IDS = ["SD-032c", "SD-021"]

C2_threshold = 0.3
C3_threshold = 10
C4_threshold = 0.02
PASS_MIN_SEEDS = 2

HARM_OBS_DIM = 51
HARM_OBS_A_DIM = 7
Z_HARM_DIM = 32
Z_HARM_A_DIM = 16

SEEDS = [42, 43, 44]
TRAIN_EPISODES = 150
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200
LR = 1e-3
WF_LR = 1e-3

CONDITIONS = ["AIC_ON", "AIC_DRIVE_ABLATED", "LEGACY", "CONTROL"]


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
        # SD-054 reef enrichment -- breaks monostrategy on 10x10 grid
        reef_enabled=True,
        n_reef_patches=3,
        reef_patch_radius=2,
        hazard_food_attraction=0.7,
    )


def make_config(condition: str, env: CausalGridWorldV2) -> REEConfig:
    from ree_core.utils.config import HeartbeatConfig
    hb = HeartbeatConfig(beta_gate_bistable=True)

    use_aic = condition in ("AIC_ON", "AIC_DRIVE_ABLATED")
    descending = condition != "CONTROL"
    drive_protect_weight = 0.0 if condition == "AIC_DRIVE_ABLATED" else 1.0

    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
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
        use_aic_analog=use_aic,
        aic_baseline_alpha=0.02,
        aic_drive_coupling=1.0,
        aic_base_attenuation=0.5,
        aic_drive_protect_weight=drive_protect_weight,
        heartbeat=hb,
    )


def run_training(agent: REEAgent, enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                 env: CausalGridWorldV2, device, n_eps: int, seed: int, condition: str,
                 n_steps: int = STEPS_PER_EPISODE):
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
            if harm_obs_a is not None:
                harm_obs_a_t = harm_obs_a.to(device)
                enc_a(harm_obs_a_t.unsqueeze(0))

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

            opt.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            harm_loss = F.mse_loss(prox_head(z_harm_s), harm_obs[-1:].unsqueeze(0))
            total = pred_loss + harm_loss
            if total.requires_grad:
                total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()

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
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            z_world_prev = z_world_curr
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 25 == 0 or ep == n_eps - 1:
            print(f"  [train] seed={seed} cond={condition} ep {ep+1}/{n_eps}", flush=True)


def eval_commitment_attenuation(agent: REEAgent, enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                                env: CausalGridWorldV2, device, n_eps: int,
                                n_steps: int = STEPS_PER_EPISODE) -> Dict:
    z_s_committed: List[float] = []
    z_s_uncommitted: List[float] = []
    z_a_committed: List[float] = []
    z_a_uncommitted: List[float] = []
    aic_saliences: List[float] = []
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

            if agent.aic is not None:
                aic_saliences.append(float(agent.aic.aic_salience))

            is_committed = agent.e3._committed_trajectory is not None
            action_idx = int(action.argmax(dim=-1).item())
            _, _, done, _, obs_dict = env.step(action_idx)

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

    return {
        "z_harm_s_ratio": safe_ratio(z_s_committed, z_s_uncommitted),
        "z_harm_a_ratio": safe_ratio(z_a_committed, z_a_uncommitted),
        "n_committed_steps": total_committed,
        "z_harm_s_mean_committed": float(np.mean(z_s_committed)) if z_s_committed else 0.0,
        "z_harm_s_mean_uncommitted": float(np.mean(z_s_uncommitted)) if z_s_uncommitted else 0.0,
        "z_harm_a_mean_committed": float(np.mean(z_a_committed)) if z_a_committed else 0.0,
        "z_harm_a_mean_uncommitted": float(np.mean(z_a_uncommitted)) if z_a_uncommitted else 0.0,
        "aic_salience_mean": float(np.mean(aic_saliences)) if aic_saliences else 0.0,
        "aic_salience_max": float(np.max(aic_saliences)) if aic_saliences else 0.0,
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_train = 2 if dry_run else TRAIN_EPISODES
    n_eval = 1 if dry_run else EVAL_EPISODES
    n_steps = 5 if dry_run else STEPS_PER_EPISODE

    condition_results = {}
    for condition in CONDITIONS:
        print(f"Seed {seed} Condition {condition}", flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_env(seed)
        cfg = make_config(condition, env)
        agent = REEAgent(cfg)
        enc_s = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM).to(device)
        enc_a = AffectiveHarmEncoder(harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM).to(device)

        run_training(agent, enc_s, enc_a, env, device, n_train, seed, condition, n_steps=n_steps)
        rv = agent.e3._running_variance
        print(f"  {condition}: rv_after_train={rv:.4f} committed_flag={rv < agent.e3.commit_threshold}", flush=True)
        metrics = eval_commitment_attenuation(agent, enc_s, enc_a, env, device, n_eval, n_steps=n_steps)
        condition_results[condition] = metrics
        print(
            f"  {condition}: z_s_ratio={metrics['z_harm_s_ratio']:.4f} "
            f"z_a_ratio={metrics['z_harm_a_ratio']:.4f} "
            f"committed={metrics['n_committed_steps']} "
            f"aic_sal_mean={metrics['aic_salience_mean']:.4f}",
            flush=True,
        )
        arm_pass = metrics["n_committed_steps"] >= C3_threshold
        print(f"verdict: {'PASS' if arm_pass else 'FAIL'}", flush=True)

    aic_on = condition_results["AIC_ON"]
    aic_abl = condition_results["AIC_DRIVE_ABLATED"]
    legacy = condition_results["LEGACY"]
    ctrl = condition_results["CONTROL"]

    c1_pass = aic_on["z_harm_s_ratio"] < ctrl["z_harm_s_ratio"]
    c2_pass = abs(aic_on["z_harm_a_ratio"] - ctrl["z_harm_a_ratio"]) < C2_threshold
    c3_pass = aic_on["n_committed_steps"] >= C3_threshold
    c4_pass = abs(aic_on["z_harm_s_ratio"] - aic_abl["z_harm_s_ratio"]) > C4_threshold
    c5_pass = aic_on["aic_salience_mean"] > 0.0
    seed_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass

    print(f"-> seed {seed}: {'PASS' if seed_pass else 'FAIL'} "
          f"C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass} C5={c5_pass}", flush=True)

    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "z_harm_s_ratio_aic_on": aic_on["z_harm_s_ratio"],
        "z_harm_s_ratio_aic_drive_ablated": aic_abl["z_harm_s_ratio"],
        "z_harm_s_ratio_legacy": legacy["z_harm_s_ratio"],
        "z_harm_s_ratio_control": ctrl["z_harm_s_ratio"],
        "z_harm_a_ratio_aic_on": aic_on["z_harm_a_ratio"],
        "z_harm_a_ratio_control": ctrl["z_harm_a_ratio"],
        "n_committed_aic_on": aic_on["n_committed_steps"],
        "n_committed_legacy": legacy["n_committed_steps"],
        "aic_salience_mean_aic_on": aic_on["aic_salience_mean"],
        "c1_attenuation_present": c1_pass,
        "c2_stream_selective": c2_pass,
        "c3_commits_exist": c3_pass,
        "c4_drive_dependence_structural": c4_pass,
        "c5_aic_salience_nontrivial": c5_pass,
        "condition_results": condition_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        "v3_exq_325d_sd032c_aic_descending_modulation_dry" if args.dry_run
        else f"v3_exq_325d_sd032c_aic_descending_modulation_{timestamp}_v3"
    )
    print(f"EXQ-325d start: {run_id}", flush=True)

    seeds_to_run = SEEDS[:1] if args.dry_run else SEEDS
    per_seed = [run_seed(s, dry_run=args.dry_run) for s in seeds_to_run]
    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-325d {outcome} ===", flush=True)
    print(f"Seeds pass: {seeds_passing}/{len(seeds_to_run)}", flush=True)
    for r in per_seed:
        s = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {s} "
            f"s_aic={r['z_harm_s_ratio_aic_on']:.4f} "
            f"s_abl={r['z_harm_s_ratio_aic_drive_ablated']:.4f} "
            f"s_ctl={r['z_harm_s_ratio_control']:.4f} "
            f"sal={r['aic_salience_mean_aic_on']:.3f}",
            flush=True,
        )

    evidence_direction = "supports" if experiment_passes else "does_not_support"
    per_claim = {
        "SD-032c": evidence_direction,
        "SD-021": evidence_direction,
    }

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": "evidence",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": per_claim,
        "supersedes": "v3_exq_325d_sd032c_aic_descending_modulation",
        "registered_thresholds": {
            "C1_attenuation_present": "aic_on < control",
            "C2_stream_selectivity": C2_threshold,
            "C3_committed_steps": C3_threshold,
            "C4_drive_dependence_abs_delta": C4_threshold,
            "C5_aic_salience_gt": 0.0,
            "seeds_needed": PASS_MIN_SEEDS,
        },
        "outcome": outcome,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "per_seed_results": per_seed,
        "seeds_passing": seeds_passing,
        "experiment_passes": experiment_passes,
        "fix_notes": (
            "EXQ-325c was SIGTERM-killed at 33 min on ree-cloud-1 (external termination, "
            "not a code crash); compute budget was 8.6x under estimated_minutes. This "
            "iteration reduces compute (3 seeds x 4 conditions x 150 train + 30 eval) "
            "and pins machine_affinity=DLAPTOP-4.local to avoid shared-vCPU preemption. "
            "Mechanism under test and pass criteria are unchanged; PASS threshold is "
            ">=2/3 seeds (was >=3/5) to match reduced seed count."
        ),
    }

    if not args.dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
        )
        out_path = write_flat_manifest(
            output,
            out_dir,
            dry_run=False,
            config=output.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Result written to: {out_path}", flush=True)
    else:
        print(f"[dry-run] outcome={outcome} (no manifest written)", flush=True)


if __name__ == "__main__":
    main()
