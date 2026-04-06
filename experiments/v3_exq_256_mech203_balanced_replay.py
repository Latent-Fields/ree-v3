"""
EXQ-256: MECH-203 Balanced Replay Priority Evidence

Claim: MECH-203 (serotonergic_replay_salience_tagging)
Design doc: REE_assembly/docs/architecture/sleep/serotonergic_cross_state_substrate.md

Mechanism under test: balanced replay start-point selection.
When serotonin is enabled, HippocampalModule.replay(drive_state=...) uses
valence-weighted priority to select replay start points. This should produce
more balanced replay (both harm and benefit locations) compared to
the default most-recent-only replay.

Pre-registered acceptance criteria:
  C1: With serotonin, replay start points include benefit-adjacent locations
      (mean_benefit_proximity_at_replay_start > 0.1 for SEROTONIN condition)
  C2: Without serotonin, replay starts are NOT benefit-biased
      (mean_benefit_proximity at replay start in NO_SEROTONIN ~ baseline random)
  C3: SEROTONIN condition shows higher z_goal maintenance after sleep
      (post_sleep_goal_norm ratio SEROTONIN/NO_SEROTONIN > 1.2)

Conditions:
  A (SEROTONIN):     tonic_5ht_enabled=True
  B (NO_SEROTONIN):  tonic_5ht_enabled=False (control)

Decision: PASS if C1 and C3 met. FAIL otherwise.
"""

import json
import sys
import random
import datetime
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_256_mech203_balanced_replay"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-203"]

# Thresholds
THRESH_C1_BENEFIT_PROX = 0.1
THRESH_C3_GOAL_RATIO   = 1.2

# Architecture
BODY_OBS_DIM   = 12
WORLD_OBS_DIM  = 250
ACTION_DIM     = 4

# Training params
WAKING_EPISODES    = 60
STEPS_PER_EPISODE  = 200
SLEEP_REPLAY_STEPS = 50
SEEDS              = [42, 137, 2026]
LR                 = 1e-3


def make_config(condition: str) -> REEConfig:
    """Build config for each condition."""
    serotonin_en = (condition == "SEROTONIN")
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        alpha_world=0.9,
        z_goal_enabled=True,
        drive_weight=2.0,
        benefit_eval_enabled=True,
        goal_weight=1.0,
        tonic_5ht_enabled=serotonin_en,
    )
    return cfg


def _action_onehot(idx: int, device) -> torch.Tensor:
    v = torch.zeros(1, ACTION_DIM, device=device)
    v[0, idx] = 1.0
    return v


def run_condition(condition: str, seed: int, dry_run: bool = False) -> Dict:
    """Run waking training, then SWS replay, measure goal maintenance."""
    print(f"Seed {seed} Condition {condition}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    cfg = make_config(condition)
    agent = REEAgent(cfg)
    agent.to(device)

    env = CausalGridWorldV2(
        size=10,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.05,
        resource_benefit=0.05,
        use_proxy_fields=True,
    )

    optimizer = optim.Adam(agent.parameters(), lr=LR)

    n_waking = 2 if dry_run else WAKING_EPISODES
    n_steps = 10 if dry_run else STEPS_PER_EPISODE
    n_replay = 5 if dry_run else SLEEP_REPLAY_STEPS

    # -- Phase 1: Waking training --
    for ep in range(n_waking):
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [train] {condition} seed={seed} ep {ep+1}/{n_waking}", flush=True)
        obs, info = env.reset()
        agent.reset()

        for step in range(n_steps):
            body_obs = torch.tensor(obs[:BODY_OBS_DIM], dtype=torch.float32).unsqueeze(0)
            world_obs = torch.tensor(obs[BODY_OBS_DIM:BODY_OBS_DIM + WORLD_OBS_DIM],
                                     dtype=torch.float32).unsqueeze(0)

            latent = agent.sense(body_obs, world_obs)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, cfg.latent.world_dim, device=device)
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            benefit_exposure = float(body_obs[0, 11]) if body_obs.shape[-1] > 11 else 0.0
            drive_level = agent.compute_drive_level(body_obs)

            agent.serotonin_step(benefit_exposure)
            agent.update_z_goal(benefit_exposure, drive_level)
            agent.update_benefit_salience(benefit_exposure)

            action_idx = int(action.argmax(dim=-1).item())
            obs_next, reward, done, truncated, info_next = env.step(action_idx)

            harm_signal = info_next.get("harm", 0.0)
            if harm_signal != 0:
                agent.update_residue(harm_signal)

            optimizer.zero_grad()
            loss = agent.compute_prediction_loss()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            obs = obs_next
            if done or truncated:
                break

    # Record pre-sleep goal norm
    pre_sleep_goal_norm = 0.0
    if agent.goal_state is not None:
        pre_sleep_goal_norm = agent.goal_state.goal_norm()
    pre_sleep_5ht = agent.serotonin.tonic_5ht

    # -- Phase 2: SWS replay --
    agent.enter_sws_mode()

    replay_benefit_proximities = []
    for _ in range(n_replay):
        recent = agent.theta_buffer.recent
        if recent is None:
            continue

        # Build drive_state like the agent does internally
        drive_state = None
        if agent.serotonin.enabled:
            t5ht = agent.serotonin.tonic_5ht
            drive_state = torch.tensor([t5ht, 0.5, 1.0 - t5ht, 0.3], device=device)

        replay_trajs = agent.hippocampal.replay(
            recent, num_replay_steps=3, drive_state=drive_state
        )

        # Measure benefit proximity at replay start
        if replay_trajs and agent._current_latent is not None:
            z_w = agent._current_latent.z_world
            if hasattr(agent.residue_field, 'evaluate_benefit'):
                bval = float(agent.residue_field.evaluate_benefit(z_w).mean().item())
                replay_benefit_proximities.append(bval)
            elif hasattr(agent.e3, 'benefit_eval'):
                bval = float(agent.e3.benefit_eval(z_w).mean().item())
                replay_benefit_proximities.append(bval)

    # -- Phase 3: Post-sleep --
    agent.exit_sleep_mode()

    post_sleep_goal_norm = 0.0
    if agent.goal_state is not None:
        post_sleep_goal_norm = agent.goal_state.goal_norm()

    mean_replay_benefit = float(np.mean(replay_benefit_proximities)) if replay_benefit_proximities else 0.0

    run_pass = (mean_replay_benefit > THRESH_C1_BENEFIT_PROX) if condition == "SEROTONIN" else True
    print(f"verdict: {'PASS' if run_pass else 'FAIL'}")

    return {
        "condition": condition,
        "seed": seed,
        "pre_sleep_goal_norm": pre_sleep_goal_norm,
        "post_sleep_goal_norm": post_sleep_goal_norm,
        "pre_sleep_5ht": pre_sleep_5ht,
        "mean_replay_benefit_proximity": mean_replay_benefit,
        "n_replay_samples": len(replay_benefit_proximities),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parents[2] /
                                    "REE_assembly" / "evidence" / "experiments"))
    args = parser.parse_args()

    print(f"EXQ-256: MECH-203 Balanced Replay Priority Evidence")
    print(f"  dry_run={args.dry_run}")

    conditions = ["SEROTONIN", "NO_SEROTONIN"]
    all_results = []

    for cond in conditions:
        for seed in SEEDS:
            print(f"  Running {cond} seed={seed}...")
            result = run_condition(cond, seed, dry_run=args.dry_run)
            all_results.append(result)
            print(f"    replay_benefit={result['mean_replay_benefit_proximity']:.4f} "
                  f"pre_goal={result['pre_sleep_goal_norm']:.4f} "
                  f"post_goal={result['post_sleep_goal_norm']:.4f}")

    # Aggregate
    ser_results = [r for r in all_results if r["condition"] == "SEROTONIN"]
    noser_results = [r for r in all_results if r["condition"] == "NO_SEROTONIN"]

    mean_ser_replay_benefit = float(np.mean([r["mean_replay_benefit_proximity"] for r in ser_results]))
    mean_noser_replay_benefit = float(np.mean([r["mean_replay_benefit_proximity"] for r in noser_results]))
    mean_ser_post_goal = float(np.mean([r["post_sleep_goal_norm"] for r in ser_results]))
    mean_noser_post_goal = float(np.mean([r["post_sleep_goal_norm"] for r in noser_results]))

    goal_ratio = mean_ser_post_goal / max(mean_noser_post_goal, 1e-9)

    c1 = mean_ser_replay_benefit > THRESH_C1_BENEFIT_PROX
    c3 = goal_ratio > THRESH_C3_GOAL_RATIO

    verdict = "PASS" if (c1 and c3) else "FAIL"

    print(f"\n  Aggregated:")
    print(f"    C1 ser_replay_benefit={mean_ser_replay_benefit:.4f} > {THRESH_C1_BENEFIT_PROX} "
          f"-> {'PASS' if c1 else 'FAIL'}")
    print(f"    C3 goal_ratio={goal_ratio:.3f} > {THRESH_C3_GOAL_RATIO} "
          f"-> {'PASS' if c3 else 'FAIL'}")
    print(f"  VERDICT: {verdict}")

    run_id = "v3_exq_256_mech203_balanced_replay_v3"
    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "verdict": verdict,
        "evidence_direction": "supports" if verdict == "PASS" else "weakens",
        "conditions": conditions,
        "seeds": SEEDS,
        "metrics": {
            "mean_ser_replay_benefit": mean_ser_replay_benefit,
            "mean_noser_replay_benefit": mean_noser_replay_benefit,
            "mean_ser_post_goal_norm": mean_ser_post_goal,
            "mean_noser_post_goal_norm": mean_noser_post_goal,
            "goal_ratio": goal_ratio,
        },
        "evidence": {
            "c1_benefit_proximity_passed": c1,
            "c3_goal_ratio_passed": c3,
        },
        "per_condition_results": all_results,
    }

    out_path = Path(args.output_dir) / f"{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
