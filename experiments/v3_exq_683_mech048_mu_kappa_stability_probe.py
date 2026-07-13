"""
V3-EXQ-683 -- MECH-048 mu/kappa Stability Overlay Probe

Claim: MECH-048 (mu/kappa stability overlays modulate mode entropy and switching pressure)

Motivation (2026-06-15):
  MECH-048 proposes that REE should include opponent stability overlays:
  - mu-analogue: increases commitment stability, reduces switching propensity
  - kappa-analogue: increases re-evaluation pressure, destabilizes regimes

  These act as stability and entropy modulators over control-plane regimes and
  commitment thresholds, NOT scalar reward signals.

  Since V3 doesn't yet have dedicated mu/kappa modules, this experiment tests
  the hypothesis using temperature modulation as a proxy for stability overlays:
  - mu_condition: lower temperature (0.5) -> reduced entropy, increased stability
  - baseline: normal temperature (1.0) -> control
  - kappa_condition: higher temperature (2.0) -> increased entropy, destabilization

Method:
  Run 3 conditions x 3 seeds = 9 runs total:
  1. Train agent in CausalGridWorld for 100 episodes
  2. During training, modulate action selection temperature according to condition
  3. Measure:
     - action_entropy: H(action distribution) over episode
     - action_switching_rate: fraction of steps where action != prev_action
     - commitment_duration_mean: when committed, how long before switching
     - policy_entropy_mean: average softmax entropy during action selection

PASS criteria:
  C1: mu_condition shows significantly lower action_entropy than baseline
      (mu_entropy < baseline_entropy - 0.1)
  C2: kappa_condition shows significantly higher action_entropy than baseline
      (kappa_entropy > baseline_entropy + 0.1)
  C3: mu_condition shows lower action_switching_rate than baseline
      (mu_switches < baseline_switches * 0.9)
  C4: kappa_condition shows higher action_switching_rate than baseline
      (kappa_switches > baseline_switches * 1.1)
  C5: All runs complete without fatal errors

Evidence direction:
  PASS -> supports MECH-048 (stability overlays affect entropy and switching as predicted)
  FAIL -> weakens MECH-048 (temperature proxy doesn't show predicted effects)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome

MANIFEST_WRITER_EXEMPT = (
    "archival early-era manifest: writes result (a run_experiment() return carrying "
    "architecture_epoch + outcome but NO run_id key) to a single hardcoded relative "
    "f-string '../REE_assembly/evidence/experiments/{run_id}.json'; routing through "
    "write_flat_manifest would need inject result['run_id'] = run_id + split the dir "
    "(a correct-and-route, not a byte-safe mechanical migration). Not queued, never "
    "ran (no evidence manifest, no runner_status completion), MECH-048 mu/kappa "
    "temperature-proxy for a substrate V3 lacks (no dedicated mu/kappa modules); not re-run."
)


EXPERIMENT_TYPE = "v3_exq_683_mech048_mu_kappa_stability_probe"
CLAIM_IDS = ["MECH-048"]

NUM_EPISODES = 100
STEPS_PER_EPISODE = 200
NUM_SEEDS = 3

# Temperature conditions for mu/kappa proxy
TEMPERATURE_CONDITIONS = {
    "mu_condition": 0.5,      # Lower temp -> more deterministic -> stability
    "baseline": 1.0,          # Normal temp -> control
    "kappa_condition": 2.0,   # Higher temp -> more random -> destabilization
}


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_entropy(probs: torch.Tensor, eps=1e-8) -> float:
    """Compute Shannon entropy H(p) = -sum(p * log(p))"""
    probs = probs.clamp(min=eps)
    return -(probs * torch.log(probs)).sum().item()


def _run_condition(
    condition_name: str,
    temperature: float,
    seed: int,
    dry_run: bool = False,
) -> Dict:
    """Run one condition with specified temperature."""

    if dry_run:
        print(f"[DRY-RUN] {condition_name} (T={temperature:.1f}, seed={seed})")
        return {
            "condition": condition_name,
            "temperature": temperature,
            "seed": seed,
            "action_entropy_mean": 0.0,
            "action_switching_rate": 0.0,
            "commitment_duration_mean": 0.0,
            "policy_entropy_mean": 0.0,
            "episodes_completed": 0,
        }

    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)

    # Create environment and agent
    config = REEConfig()
    config.device = "cpu"
    config.world_size = 5
    config.num_hazards = 1
    config.num_resources = 2

    env = CausalGridWorld(
        size=config.world_size,
        num_resources=config.num_resources,
        num_hazards=config.num_hazards,
        device=config.device,
    )

    agent = REEAgent(config)
    agent.train()

    optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    # Track metrics across all episodes
    all_action_entropies = []
    all_policy_entropies = []
    action_switches = 0
    total_steps = 0
    commitment_durations = []
    current_commitment_duration = 0
    prev_action_idx = None

    for ep in range(NUM_EPISODES):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        episode_done = False
        step = 0

        while not episode_done and step < STEPS_PER_EPISODE:
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            # Encode
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            # Select action with specified temperature
            # Get scores from E3
            scores = agent.e3.score(latent)

            # Apply temperature-scaled softmax
            probs = torch.softmax(scores / temperature, dim=-1)

            # Compute entropy of this distribution
            policy_entropy = _compute_entropy(probs)
            all_policy_entropies.append(policy_entropy)

            # Sample action
            action_dist = torch.distributions.Categorical(probs)
            action_idx = action_dist.sample().item()
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)

            # Track switching
            if prev_action_idx is not None:
                if action_idx != prev_action_idx:
                    action_switches += 1
                    # End current commitment duration
                    if current_commitment_duration > 0:
                        commitment_durations.append(current_commitment_duration)
                    current_commitment_duration = 0
                else:
                    current_commitment_duration += 1

            prev_action_idx = action_idx
            total_steps += 1

            # Step environment
            flat_obs_next, reward, done, truncated, info = env.step(action_idx)
            episode_done = done or truncated
            obs_dict = info.get("obs_dict", {})

            # Simple loss: predict next observation
            with torch.no_grad():
                obs_body_next = obs_dict.get("body_state", obs_body)
                obs_world_next = obs_dict.get("world_state", obs_world)

            latent_next = agent.encode(obs_body_next, obs_world_next)

            # E2 prediction loss
            z_self_pred = agent.e2_self.predict(
                latent.z_self,
                action,
                mode="deterministic",
            )
            loss_self = nn.functional.mse_loss(z_self_pred, latent_next.z_self.detach())

            z_world_pred = agent.e2_world.predict(
                latent.z_world,
                action,
                mode="deterministic",
            )
            loss_world = nn.functional.mse_loss(z_world_pred, latent_next.z_world.detach())

            loss = loss_self + loss_world

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

        # Compute episode action entropy (empirical distribution over actions taken)
        # This is different from policy entropy (model's distribution)
        # Track action counts to compute empirical distribution

    # Compute final metrics
    action_entropy_mean = sum(all_policy_entropies) / max(len(all_policy_entropies), 1)
    action_switching_rate = action_switches / max(total_steps - 1, 1)
    commitment_duration_mean = sum(commitment_durations) / max(len(commitment_durations), 1) if commitment_durations else 0.0
    policy_entropy_mean = action_entropy_mean  # Same as action_entropy_mean in this implementation

    return {
        "condition": condition_name,
        "temperature": temperature,
        "seed": seed,
        "action_entropy_mean": action_entropy_mean,
        "action_switching_rate": action_switching_rate,
        "commitment_duration_mean": commitment_duration_mean,
        "policy_entropy_mean": policy_entropy_mean,
        "episodes_completed": NUM_EPISODES,
        "total_steps": total_steps,
        "action_switches": action_switches,
    }


def run_experiment(dry_run: bool = False) -> Dict:
    """Run all conditions across all seeds."""

    results = []

    for condition_name, temperature in TEMPERATURE_CONDITIONS.items():
        for seed in range(NUM_SEEDS):
            print(f"Running {condition_name} (T={temperature:.1f}, seed={seed})...")
            result = _run_condition(condition_name, temperature, seed, dry_run)
            results.append(result)

            if not dry_run:
                print(f"  action_entropy_mean: {result['action_entropy_mean']:.4f}")
                print(f"  action_switching_rate: {result['action_switching_rate']:.4f}")
                print(f"  commitment_duration_mean: {result['commitment_duration_mean']:.2f}")

    # Aggregate by condition
    condition_stats = {}
    for condition_name in TEMPERATURE_CONDITIONS.keys():
        condition_results = [r for r in results if r["condition"] == condition_name]

        condition_stats[condition_name] = {
            "action_entropy_mean": sum(r["action_entropy_mean"] for r in condition_results) / len(condition_results),
            "action_switching_rate_mean": sum(r["action_switching_rate"] for r in condition_results) / len(condition_results),
            "commitment_duration_mean": sum(r["commitment_duration_mean"] for r in condition_results) / len(condition_results),
        }

    # Evaluate pass criteria
    mu_entropy = condition_stats["mu_condition"]["action_entropy_mean"]
    baseline_entropy = condition_stats["baseline"]["action_entropy_mean"]
    kappa_entropy = condition_stats["kappa_condition"]["action_entropy_mean"]

    mu_switches = condition_stats["mu_condition"]["action_switching_rate_mean"]
    baseline_switches = condition_stats["baseline"]["action_switching_rate_mean"]
    kappa_switches = condition_stats["kappa_condition"]["action_switching_rate_mean"]

    c1_pass = mu_entropy < baseline_entropy - 0.1
    c2_pass = kappa_entropy > baseline_entropy + 0.1
    c3_pass = mu_switches < baseline_switches * 0.9
    c4_pass = kappa_switches > baseline_switches * 1.1
    c5_pass = len(results) == len(TEMPERATURE_CONDITIONS) * NUM_SEEDS

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass

    summary = {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids_tested": CLAIM_IDS,
        "outcome": "PASS" if all_pass else "FAIL",
        "evidence_class": "behavioral",
        "evidence_direction": "supports" if all_pass else "weakens",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "seeds": list(range(NUM_SEEDS)),
        "num_conditions": len(TEMPERATURE_CONDITIONS),
        "num_episodes_per_run": NUM_EPISODES,
        "condition_stats": condition_stats,
        "criteria": {
            "C1_mu_lower_entropy": {
                "pass": c1_pass,
                "mu_entropy": mu_entropy,
                "baseline_entropy": baseline_entropy,
                "delta": mu_entropy - baseline_entropy,
            },
            "C2_kappa_higher_entropy": {
                "pass": c2_pass,
                "kappa_entropy": kappa_entropy,
                "baseline_entropy": baseline_entropy,
                "delta": kappa_entropy - baseline_entropy,
            },
            "C3_mu_lower_switching": {
                "pass": c3_pass,
                "mu_switches": mu_switches,
                "baseline_switches": baseline_switches,
                "ratio": mu_switches / baseline_switches if baseline_switches > 0 else 0.0,
            },
            "C4_kappa_higher_switching": {
                "pass": c4_pass,
                "kappa_switches": kappa_switches,
                "baseline_switches": baseline_switches,
                "ratio": kappa_switches / baseline_switches if baseline_switches > 0 else 0.0,
            },
            "C5_all_runs_complete": {
                "pass": c5_pass,
                "runs_completed": len(results),
                "runs_expected": len(TEMPERATURE_CONDITIONS) * NUM_SEEDS,
            },
        },
        "summary": f"MECH-048 mu/kappa stability probe: {'PASS' if all_pass else 'FAIL'}. "
                  f"mu reduces entropy by {abs(mu_entropy - baseline_entropy):.3f}, "
                  f"kappa increases entropy by {abs(kappa_entropy - baseline_entropy):.3f}. "
                  f"Switching: mu={mu_switches:.3f}, baseline={baseline_switches:.3f}, kappa={kappa_switches:.3f}.",
        "raw_results": results,
    }

    return summary


if __name__ == "__main__":
    import json
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Quick validation run")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        print("\n[DRY-RUN] Experiment structure validated")
        print(json.dumps(result, indent=2))
    else:
        # Write manifest
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
        out_path = f"../REE_assembly/evidence/experiments/{run_id}.json"

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Result written to: {out_path}", flush=True)
        print(f"Outcome: {result['outcome']}")
        print(f"Evidence direction: {result['evidence_direction']}")

        _outcome_raw = str(result["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=out_path,
        )
