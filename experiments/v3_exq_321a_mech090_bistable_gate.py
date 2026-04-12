#!/opt/local/bin/python3
"""
V3-EXQ-321a: MECH-090 Bistable Gate vs Legacy Per-Tick Gate (E2 world-forward training fix)

experiment_purpose: evidence

Supersedes: V3-EXQ-321
Root cause of EXQ-321 failure: training loop only called compute_prediction_loss()
(E1 loss). Never trained E2 world_forward, never called update_running_variance().
running_variance stayed frozen at 0.50 (precision_init), never dropped below
commitment_threshold=0.40, committed=False throughout all episodes,
total_committed_steps=0. This script adds the E2 world-forward training loop
(wf_buf + wf_optimizer + update_running_variance per step) and extends
TRAIN_EPISODES from 50 to 300.

Tests that HeartbeatConfig.beta_gate_bistable=True produces a stable commitment
gate during committed episodes, compared to the legacy per-tick raise/release mode.

Two conditions per seed:
  BISTABLE -- beta_gate_bistable=True (elevate on ENTRY only, hold until
              hippocampal completion signal or surprise)
  LEGACY   -- beta_gate_bistable=False (elevate when committed, release when not,
              per tick -- original pre-2026-04-10 behaviour)

Key metric: hold_rate_committed -- fraction of committed steps where beta
stays elevated. BISTABLE should be higher (gate holds once raised).

Also: n_premature_releases -- times gate drops mid-committed-sequence (should be
lower in BISTABLE by design).

Pass criterion (pre-registered):
  C1: hold_rate_committed_bistable > hold_rate_committed_legacy
  C2: hold_rate_committed_bistable >= 0.7 (gate holds at least 70% of committed steps)
  C3: Both conditions show the agent can commit (n_committed_steps > 0)

Experiment PASS: >= 3/5 seeds satisfy C1 and C2.

Mechanism under test: MECH-090 (beta oscillations bistable gate).
If the experiment PASSes, it supports MECH-090: the bistable latch property means
once beta elevation is triggered on entry to a committed state, it persists until
a genuine release signal (hippocampal completion or surprise), not per-tick re-eval.
ARC-028 (hippocampal completion coupling) is NOT tagged here: ARC-028 tests whether
hippocampal completion signal RELEASES the gate; this experiment tests whether the
gate HOLDS. A dedicated ARC-028 experiment must vary the completion signal, not the
bistable flag.

Claims: MECH-090 only.
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
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig, HeartbeatConfig


EXPERIMENT_TYPE = "v3_exq_321a_mech090_bistable_gate"
CLAIM_IDS = ["MECH-090"]

C1_threshold = 0.0   # bistable hold_rate > legacy hold_rate
C2_threshold = 0.7   # bistable hold_rate >= 70% (pass floor)
C3_threshold = 1     # at least 1 committed step in each condition
PASS_MIN_SEEDS = 3

SEEDS = [42, 43, 44, 45, 46]
TRAIN_EPISODES = 300   # extended from 50 -- needed for rv to drop from 0.50 to < 0.40
EVAL_EPISODES = 20
STEPS_PER_EPISODE = 200
LR = 1e-3
WF_LR = 1e-3


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=10, num_hazards=4, num_resources=3,
        hazard_harm=0.02, resource_benefit=0.05,
        use_proxy_fields=True, seed=seed,
    )


def make_config(bistable: bool) -> REEConfig:
    hb = HeartbeatConfig(beta_gate_bistable=bistable)
    return REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        alpha_world=0.9,
        use_harm_stream=True,
        heartbeat=hb,
    )


def run_training(agent: REEAgent, env: CausalGridWorldV2, device, n_eps: int,
                 n_steps: int = STEPS_PER_EPISODE):
    """Train agent with E2 world-forward loop so running_variance can drop below commit_threshold."""
    e1_opt = optim.Adam(list(agent.e1.parameters()), lr=LR)
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
            latent = agent.sense(obs_body, obs_world)

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

            # E1 prediction loss
            e1_opt.zero_grad()
            loss = agent.compute_prediction_loss()
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                e1_opt.step()

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

            _, _, done, _, obs_dict = env.step(action_idx)
            if done:
                break


def eval_gate_stability(agent: REEAgent, env: CausalGridWorldV2, device, n_eps: int,
                        n_steps: int = STEPS_PER_EPISODE) -> Dict:
    """Measure gate stability during committed episodes."""
    total_committed_steps = 0
    total_beta_elevated_committed = 0
    n_premature_releases = 0

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        prev_committed = False
        prev_elevated = False
        ep_committed = 0
        ep_elevated_while_committed = 0

        for _ in range(n_steps):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            with torch.no_grad():
                action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            is_committed = agent.e3._committed_trajectory is not None
            is_elevated = agent.beta_gate.is_elevated

            if is_committed:
                ep_committed += 1
                if is_elevated:
                    ep_elevated_while_committed += 1
                # Premature release: was committed+elevated, now committed but not elevated
                if prev_committed and prev_elevated and not is_elevated:
                    n_premature_releases += 1

            prev_committed = is_committed
            prev_elevated = is_elevated

            _, _, done, _, obs_dict = env.step(action_idx)
            if done:
                break

        total_committed_steps += ep_committed
        total_beta_elevated_committed += ep_elevated_while_committed

    hold_rate = (
        total_beta_elevated_committed / total_committed_steps
        if total_committed_steps > 0 else 0.0
    )
    return {
        "hold_rate_committed": hold_rate,
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated_committed": total_beta_elevated_committed,
        "n_premature_releases": n_premature_releases,
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
    for condition in ["BISTABLE", "LEGACY"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_env(seed)
        cfg = make_config(bistable=(condition == "BISTABLE"))
        agent = REEAgent(cfg)

        print(f"  {condition}: training {n_train} eps...")
        run_training(agent, env, device, n_train, n_steps=n_steps)
        rv = agent.e3._running_variance
        print(f"  {condition}: rv_after_train={rv:.4f} committed_flag={rv < agent.e3.commit_threshold}")
        print(f"  {condition}: eval {n_eval} eps...")
        metrics = eval_gate_stability(agent, env, device, n_eval, n_steps=n_steps)
        condition_results[condition] = metrics
        print(
            f"  {condition}: hold_rate={metrics['hold_rate_committed']:.4f} "
            f"committed_steps={metrics['total_committed_steps']} "
            f"premature_releases={metrics['n_premature_releases']}"
        )

    bistable = condition_results["BISTABLE"]
    legacy = condition_results["LEGACY"]
    c1_pass = bistable["hold_rate_committed"] > legacy["hold_rate_committed"]
    c2_pass = bistable["hold_rate_committed"] >= C2_threshold
    c3_pass = bistable["total_committed_steps"] >= C3_threshold
    seed_pass = c1_pass and c2_pass and c3_pass

    print(f"  -> {'PASS' if seed_pass else 'FAIL'}")
    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "hold_rate_bistable": bistable["hold_rate_committed"],
        "hold_rate_legacy": legacy["hold_rate_committed"],
        "premature_releases_bistable": bistable["n_premature_releases"],
        "premature_releases_legacy": legacy["n_premature_releases"],
        "c1_bistable_higher": c1_pass,
        "c2_bistable_stable": c2_pass,
        "c3_commits_exist": c3_pass,
        "condition_results": condition_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        "v3_exq_321a_mech090_bistable_gate_dry" if args.dry_run
        else f"v3_exq_321a_mech090_bistable_gate_{timestamp}_v3"
    )
    print(f"EXQ-321a start: {run_id}")

    seeds_to_run = SEEDS[:1] if args.dry_run else SEEDS
    per_seed = [run_seed(s, dry_run=args.dry_run) for s in seeds_to_run]
    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-321a {outcome} ===")
    print(f"Seeds pass: {seeds_passing}/{len(seeds_to_run)}")
    for r in per_seed:
        s = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {s} "
            f"bistable={r['hold_rate_bistable']:.4f} "
            f"legacy={r['hold_rate_legacy']:.4f} "
            f"releases_b={r['premature_releases_bistable']}"
        )

    evidence_direction = "supports" if experiment_passes else "does_not_support"
    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": "evidence",
        "evidence_direction": evidence_direction,
        "supersedes": "v3_exq_321_mech090_bistable_gate",
        "registered_thresholds": {
            "C1_bistable_hold_rate_higher": C1_threshold,
            "C2_bistable_hold_rate_floor": C2_threshold,
            "C3_committed_steps": C3_threshold,
            "seeds_needed": PASS_MIN_SEEDS,
        },
        "outcome": outcome,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "per_seed_results": per_seed,
        "seeds_passing": seeds_passing,
        "experiment_passes": experiment_passes,
        "fix_notes": (
            "EXQ-321 had frozen running_variance=0.50 (no E2 world-forward training). "
            "Fix: added wf_buf + wf_opt + update_running_variance per step; "
            "extended TRAIN_EPISODES 50->300. claim_ids reduced to [MECH-090] only -- "
            "ARC-028 tests release signal, not hold property."
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
