#!/usr/bin/env python3
"""
EXQ-244a: MECH-165 reverse replay diversity scheduler validation.

Supersedes EXQ-244 (non_contributory: proxy substrate, no exploration source material).

Design (EXP-0105 redesign):
  Phase 1 (exploration): warmup with epsilon=0.5 random exploration, generating
  diverse behavioral trajectories into exploration buffer. Measure baseline
  action_entropy. All conditions run identically in Phase 1.

  Phase 2 (consolidation + test): three conditions differ ONLY in replay mode:
    A) NO_REPLAY:       replay_diversity_enabled=False, SWS replay skipped entirely
    B) FORWARD_REPLAY:  replay_diversity_enabled=False, standard forward replay
    C) BALANCED_REPLAY: replay_diversity_enabled=True, diverse scheduler (30% reverse,
                        20% random, 50% forward)

  PASS criterion: BALANCED condition maintains action_entropy from Phase 1 better
  than FORWARD condition in >= 3/5 seeds. Specifically:
    entropy_retention = phase2_entropy / phase1_entropy
    PASS if mean(retention_BALANCED) > mean(retention_FORWARD) AND
         BALANCED > FORWARD in >= 3/5 seeds

Mechanism under test:
  MECH-165: offline replay must sample trajectory-diverse content to maintain
  multi-strategy viability. Reverse replay and balanced scheduling prevent
  monostrategy convergence in E1's world model.

claim_ids: [MECH-165]
experiment_purpose: diagnostic
architecture_epoch: ree_hybrid_guardrails_v1
"""

import json
import os
import sys
import time
import random
from collections import Counter
from typing import Dict, List

import numpy as np
import torch

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2

# ---------- Constants ----------

BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4

EXPLORATION_EPISODES = 30      # Phase 1: exploration warmup
EXPLORATION_STEPS = 200        # Steps per exploration episode
EXPLORATION_EPSILON = 0.5      # Random action probability during Phase 1

CONSOLIDATION_EPISODES = 20    # Phase 2: consolidation + test
CONSOLIDATION_STEPS = 200      # Steps per consolidation episode
NUM_SWS_CYCLES = 5             # SWS replay cycles between Phase 1 and Phase 2

SEEDS = [42, 123, 456, 789, 1024]
NUM_SEEDS = len(SEEDS)

CONDITIONS = ["NO_REPLAY", "FORWARD_REPLAY", "BALANCED_REPLAY"]

# PASS thresholds
PASS_SEED_MAJORITY = 3  # BALANCED > FORWARD in >= 3/5 seeds


def make_config(condition: str) -> REEConfig:
    """Create config for a given condition."""
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        alpha_world=0.9,
        # SHY must be on for replay diversity to work correctly (MECH-120)
        shy_enabled=True,
        shy_decay_rate=0.85,
        # MECH-165: replay diversity
        replay_diversity_enabled=(condition == "BALANCED_REPLAY"),
        reverse_replay_fraction=0.3,
        random_replay_fraction=0.2,
        exploration_buffer_len=50,
    )
    return cfg


def compute_action_entropy(action_counts: Counter) -> float:
    """Shannon entropy of action distribution."""
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in action_counts.values()]
    return -sum(p * np.log(p + 1e-10) for p in probs if p > 0)


def run_phase1_exploration(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    epsilon: float,
) -> Dict:
    """Phase 1: exploration warmup. Returns action entropy."""
    action_counts: Counter = Counter()
    total_reward = 0.0

    for ep in range(EXPLORATION_EPISODES):
        flat_obs, _ = env.reset()
        agent.reset()
        for step in range(EXPLORATION_STEPS):
            if random.random() < epsilon:
                # Random exploration action
                action_idx = random.randint(0, ACTION_DIM - 1)
                action_tensor = torch.zeros(1, ACTION_DIM)
                action_tensor[0, action_idx] = 1.0
                # Still sense to update latent state
                agent.sense_flat(flat_obs)
                # Record exploration state/action manually
                agent._record_exploration_action(action_tensor)
            else:
                action = agent.act(flat_obs)
                action_idx = int(action.argmax(-1).item()) if action.dim() > 1 else int(action.item())

            action_counts[action_idx] += 1
            flat_obs, harm, done, info, _ = env.step(action_idx)
            total_reward += harm
            if done:
                flat_obs, _ = env.reset()
                agent.reset()

    # Final flush
    agent.reset()

    entropy = compute_action_entropy(action_counts)
    return {
        "action_entropy": entropy,
        "total_reward": total_reward,
        "action_distribution": dict(action_counts),
        "exploration_buffer_size": len(agent.hippocampal._exploration_buffer),
    }


def run_sws_consolidation(agent: REEAgent, num_cycles: int, condition: str) -> Dict:
    """Run SWS consolidation cycles."""
    metrics: Dict = {"sws_cycles": num_cycles, "reverse_replayed": 0}

    for _ in range(num_cycles):
        agent.enter_sws_mode()

        # Trigger replay
        recent = agent.theta_buffer.recent
        if recent is not None:
            if condition == "NO_REPLAY":
                pass  # Skip replay entirely
            elif condition == "FORWARD_REPLAY":
                trajs = agent.hippocampal.replay(recent)
            elif condition == "BALANCED_REPLAY":
                trajs = agent.hippocampal.diverse_replay(recent, mode="auto")
                metrics["reverse_replayed"] += sum(
                    1 for t in trajs if t.is_reverse
                )

        agent.exit_sleep_mode()

    return metrics


def run_phase2_test(
    agent: REEAgent,
    env: CausalGridWorldV2,
) -> Dict:
    """Phase 2: test with no exploration (epsilon=0). Returns action entropy."""
    action_counts: Counter = Counter()
    total_reward = 0.0

    for ep in range(CONSOLIDATION_EPISODES):
        flat_obs, _ = env.reset()
        agent.reset()
        for step in range(CONSOLIDATION_STEPS):
            action = agent.act(flat_obs)
            action_idx = int(action.argmax(-1).item()) if action.dim() > 1 else int(action.item())
            action_counts[action_idx] += 1
            flat_obs, harm, done, info, _ = env.step(action_idx)
            total_reward += harm
            if done:
                flat_obs, _ = env.reset()
                agent.reset()

    entropy = compute_action_entropy(action_counts)
    return {
        "action_entropy": entropy,
        "total_reward": total_reward,
        "action_distribution": dict(action_counts),
    }


def run_condition(condition: str, seed: int) -> Dict:
    """Run one condition for one seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg = make_config(condition)
    agent = REEAgent(cfg)
    env = CausalGridWorldV2()

    # Phase 1: exploration
    phase1 = run_phase1_exploration(agent, env, seed, EXPLORATION_EPSILON)
    phase1_entropy = phase1["action_entropy"]

    # SWS consolidation
    sws = run_sws_consolidation(agent, NUM_SWS_CYCLES, condition)

    # Phase 2: test (no exploration)
    phase2 = run_phase2_test(agent, env)
    phase2_entropy = phase2["action_entropy"]

    # Compute retention
    if phase1_entropy > 0:
        entropy_retention = phase2_entropy / phase1_entropy
    else:
        entropy_retention = 0.0

    return {
        "condition": condition,
        "seed": seed,
        "phase1_entropy": phase1_entropy,
        "phase2_entropy": phase2_entropy,
        "entropy_retention": entropy_retention,
        "phase1_action_dist": phase1["action_distribution"],
        "phase2_action_dist": phase2["action_distribution"],
        "exploration_buffer_size": phase1["exploration_buffer_size"],
        "sws_metrics": sws,
    }


def main():
    t0 = time.time()
    print(f"EXQ-244a: MECH-165 replay diversity validation")
    print(f"Conditions: {CONDITIONS}")
    print(f"Seeds: {SEEDS}")
    print(f"Phase 1: {EXPLORATION_EPISODES} episodes x {EXPLORATION_STEPS} steps (epsilon={EXPLORATION_EPSILON})")
    print(f"Phase 2: {CONSOLIDATION_EPISODES} episodes x {CONSOLIDATION_STEPS} steps (greedy)")
    print()

    all_results: List[Dict] = []

    for condition in CONDITIONS:
        print(f"--- Condition: {condition} ---")
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = run_condition(condition, seed)
            all_results.append(result)
            print(f"retention={result['entropy_retention']:.3f} "
                  f"(P1={result['phase1_entropy']:.3f}, P2={result['phase2_entropy']:.3f})")

    # Aggregate per condition
    condition_stats: Dict = {}
    for cond in CONDITIONS:
        cond_results = [r for r in all_results if r["condition"] == cond]
        retentions = [r["entropy_retention"] for r in cond_results]
        condition_stats[cond] = {
            "mean_retention": float(np.mean(retentions)),
            "std_retention": float(np.std(retentions)),
            "retentions": retentions,
        }

    # PASS criterion: BALANCED retention > FORWARD retention in >= 3/5 seeds
    balanced_rets = condition_stats["BALANCED_REPLAY"]["retentions"]
    forward_rets = condition_stats["FORWARD_REPLAY"]["retentions"]

    seeds_balanced_wins = sum(
        1 for b, f in zip(balanced_rets, forward_rets) if b > f
    )
    pass_criterion = seeds_balanced_wins >= PASS_SEED_MAJORITY
    mean_balanced = condition_stats["BALANCED_REPLAY"]["mean_retention"]
    mean_forward = condition_stats["FORWARD_REPLAY"]["mean_retention"]
    mean_advantage = mean_balanced > mean_forward

    outcome = "PASS" if (pass_criterion and mean_advantage) else "FAIL"
    elapsed = time.time() - t0

    print()
    print(f"=== RESULTS ===")
    for cond in CONDITIONS:
        stats = condition_stats[cond]
        print(f"  {cond}: mean_retention={stats['mean_retention']:.4f} "
              f"+/- {stats['std_retention']:.4f}")
    print(f"  BALANCED > FORWARD: {seeds_balanced_wins}/{NUM_SEEDS} seeds "
          f"(need >= {PASS_SEED_MAJORITY})")
    print(f"  mean advantage: {mean_balanced:.4f} > {mean_forward:.4f} = {mean_advantage}")
    print(f"  Outcome: {outcome}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # Build output
    run_id = f"v3_exq_244a_mech165_replay_diversity_validation_v3"
    output = {
        "run_id": run_id,
        "experiment_type": "v3_exq_244a_mech165_replay_diversity_validation",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": ["MECH-165"],
        "experiment_purpose": "diagnostic",
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "conditions": CONDITIONS,
        "seeds": SEEDS,
        "condition_stats": condition_stats,
        "pass_criteria": {
            "seeds_balanced_wins": seeds_balanced_wins,
            "seeds_needed": PASS_SEED_MAJORITY,
            "mean_advantage": mean_advantage,
            "pass_criterion_met": pass_criterion,
        },
        "phase1_config": {
            "episodes": EXPLORATION_EPISODES,
            "steps": EXPLORATION_STEPS,
            "epsilon": EXPLORATION_EPSILON,
        },
        "phase2_config": {
            "episodes": CONSOLIDATION_EPISODES,
            "steps": CONSOLIDATION_STEPS,
        },
        "sws_cycles": NUM_SWS_CYCLES,
        "per_seed_results": all_results,
        "elapsed_seconds": elapsed,
        "supersedes": "v3_exq_244_mech165_replay_diversity_monostrategy_v3",
    }

    # Write flat JSON to evidence directory
    evidence_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
    )
    os.makedirs(evidence_dir, exist_ok=True)
    outpath = os.path.join(evidence_dir, f"{run_id}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults written to: {outpath}")


if __name__ == "__main__":
    main()
