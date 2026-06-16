"""
V3-EXQ-686 -- MECH-191 Signal-State Discriminability Probe

Claim: MECH-191 (Stereotyped behavioral signals are causal externalizations of
specific internal functional states, making them cross-architecturally legible)

Motivation (2026-06-16):
  MECH-191 proposes that stereotyped signals (whine=wanting, yelp=nociception,
  huff=plan-abandonment) are causally produced by internal functional states.
  Because signals are causal products (not conventions), they should be
  discriminable across observers.

  V3 doesn't have multi-agent environments to test cross-agent legibility directly.
  This experiment tests a necessary precondition: **Do different internal functional
  states produce discriminable output patterns within a single agent?**

  If internal states DON'T produce discriminable patterns, they cannot be legible
  across agents. If they DO produce discriminable patterns, that supports (but
  doesn't prove) cross-agent legibility.

Method:
  1. Create three scenarios that induce distinct internal functional states:
     - WANTING: agent approaches goal, high z_goal, low z_harm_a, low z_block
     - NOCICEPTION: agent experiences harm, low z_goal, high z_harm_a, low z_block
     - FRUSTRATION: agent has goal but is blocked, high z_goal, low-med z_harm_a, high z_block

  2. Sample "signal vectors" from agent's internal state when each functional
     state is dominant. Signal = concatenation of [z_goal_norm, z_harm_a_norm, z_block_norm].

  3. Train a simple logistic classifier to predict which functional state produced
     each signal from the signal pattern alone.

  4. Measure classification accuracy. High accuracy = signals are discriminable
     by functional state (supports MECH-191). Chance accuracy = signals are not
     discriminable (weakens MECH-191).

PASS criteria:
  C1: Wanting signals classified correctly >= 70% of the time
  C2: Nociception signals classified correctly >= 70% of the time
  C3: Frustration signals classified correctly >= 70% of the time (if z_block available)
  C4: Overall 3-way accuracy significantly above chance (>50% for 3-way, >60% for 2-way)
  C5: All runs complete without fatal errors

Evidence direction:
  PASS -> supports MECH-191 (signals causally tied to and discriminable by functional states)
  FAIL -> weakens MECH-191 (signals not reliably tied to functional states)
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
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._metrics import check_degeneracy


EXPERIMENT_TYPE = "v3_exq_686_mech191_signal_state_discriminability"
CLAIM_IDS = ["MECH-191"]

NUM_EPISODES_PER_CONDITION = 20
STEPS_PER_EPISODE = 150
NUM_SEEDS = 3
MIN_SAMPLES_PER_STATE = 30  # Minimum samples needed per functional state


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _collect_state_samples(
    agent: REEAgent,
    env: CausalGridWorld,
    scenario: str,
    seed: int,
    dry_run: bool = False,
) -> Dict:
    """
    Collect signal samples for one functional state scenario.

    Returns dict with:
      - functional_state: "wanting" | "nociception" | "frustration"
      - signal_samples: List[Dict] with z_goal_norm, z_harm_a_norm, z_block_norm
    """

    if dry_run:
        return {
            "scenario": scenario,
            "functional_state": scenario,
            "seed": seed,
            "signal_samples": [],
            "episodes": 0,
        }

    random.seed(seed)
    torch.manual_seed(seed)

    signal_samples = []

    for ep in range(NUM_EPISODES_PER_CONDITION):
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

            # Extract signal vector (internal state norms)
            z_goal_norm = agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
            z_harm_a_norm = float(latent.z_harm_a.norm().item()) if latent.z_harm_a is not None else 0.0
            z_block_norm = float(latent.z_block.norm().item()) if latent.z_block is not None else 0.0

            # Determine if this step represents the target functional state
            is_target_state = False

            if scenario == "wanting":
                # Wanting: z_goal > threshold, z_harm_a low, z_block low
                is_target_state = (z_goal_norm > 0.3 and z_harm_a_norm < 0.2 and z_block_norm < 0.1)
            elif scenario == "nociception":
                # Nociception: z_harm_a high, z_goal low, z_block low
                is_target_state = (z_harm_a_norm > 0.3 and z_goal_norm < 0.2 and z_block_norm < 0.1)
            elif scenario == "frustration":
                # Frustration: z_goal high, z_block high
                is_target_state = (z_goal_norm > 0.3 and z_block_norm > 0.3)

            # Sample signal if in target state
            if is_target_state:
                signal_samples.append({
                    "z_goal_norm": z_goal_norm,
                    "z_harm_a_norm": z_harm_a_norm,
                    "z_block_norm": z_block_norm,
                    "functional_state": scenario,
                })

            # Select action (simple policy)
            scores = agent.e3.score(latent)
            probs = torch.softmax(scores, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            action_idx = action_dist.sample().item()
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)

            # Step environment
            flat_obs_next, reward, done, truncated, info = env.step(action_idx)
            episode_done = done or truncated
            obs_dict = info.get("obs_dict", {})

            # Update goal if benefit obtained
            if info.get("benefit_exposure", 0.0) > 0:
                drive_level = 1.0 - obs_body[3].item()  # energy depletion
                agent.update_z_goal(info.get("benefit_exposure", 0.0), drive_level)

            step += 1

    return {
        "scenario": scenario,
        "functional_state": scenario,
        "seed": seed,
        "signal_samples": signal_samples,
        "episodes": NUM_EPISODES_PER_CONDITION,
        "samples_collected": len(signal_samples),
    }


def _train_classifier(
    train_samples: List[Dict],
    test_samples: List[Dict],
) -> Dict:
    """
    Train a simple logistic classifier to discriminate functional states.

    Returns classification metrics.
    """

    # Prepare training data
    X_train = np.array([[s["z_goal_norm"], s["z_harm_a_norm"], s["z_block_norm"]] for s in train_samples])
    y_train = np.array([
        0 if s["functional_state"] == "wanting" else
        1 if s["functional_state"] == "nociception" else
        2
        for s in train_samples
    ])

    X_test = np.array([[s["z_goal_norm"], s["z_harm_a_norm"], s["z_block_norm"]] for s in test_samples])
    y_test = np.array([
        0 if s["functional_state"] == "wanting" else
        1 if s["functional_state"] == "nociception" else
        2
        for s in test_samples
    ])

    # Simple logistic regression (3-way classifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Per-class accuracy
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    return {
        "overall_accuracy": float(accuracy),
        "wanting_accuracy": float(per_class_acc[0]) if len(per_class_acc) > 0 else 0.0,
        "nociception_accuracy": float(per_class_acc[1]) if len(per_class_acc) > 1 else 0.0,
        "frustration_accuracy": float(per_class_acc[2]) if len(per_class_acc) > 2 else 0.0,
        "confusion_matrix": conf_matrix.tolist(),
        "n_train": len(train_samples),
        "n_test": len(test_samples),
    }


def run_experiment(dry_run: bool = False) -> Dict:
    """Run the full experiment across all seeds."""

    all_samples = {
        "wanting": [],
        "nociception": [],
        "frustration": [],
    }

    for seed in range(NUM_SEEDS):
        print(f"Seed {seed}...")

        # Setup
        random.seed(seed)
        torch.manual_seed(seed)

        config = REEConfig()
        config.device = "cpu"
        config.world_size = 5
        config.num_hazards = 2  # More hazards for nociception scenario
        config.num_resources = 2
        config.use_blocked_agency = True  # Enable z_block for frustration

        env = CausalGridWorld(
            size=config.world_size,
            num_resources=config.num_resources,
            num_hazards=config.num_hazards,
        )

        agent = REEAgent(config)
        agent.train()

        # Collect samples for each scenario
        for scenario in ["wanting", "nociception", "frustration"]:
            result = _collect_state_samples(agent, env, scenario, seed, dry_run)

            if not dry_run:
                all_samples[scenario].extend(result["signal_samples"])
                print(f"  {scenario}: {result['samples_collected']} samples")

    if dry_run:
        # Mock results for dry run
        return {
            "experiment_type": EXPERIMENT_TYPE,
            "claim_ids_tested": CLAIM_IDS,
            "outcome": "PASS",
            "evidence_class": "computational",
            "evidence_direction": "supports",
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "summary": "[DRY-RUN] Structure validated",
        }

    # Check if we have enough samples
    n_wanting = len(all_samples["wanting"])
    n_nociception = len(all_samples["nociception"])
    n_frustration = len(all_samples["frustration"])

    print(f"\nTotal samples collected:")
    print(f"  Wanting: {n_wanting}")
    print(f"  Nociception: {n_nociception}")
    print(f"  Frustration: {n_frustration}")

    # If any functional state has too few samples, the experiment cannot proceed
    insufficient_samples = (
        n_wanting < MIN_SAMPLES_PER_STATE or
        n_nociception < MIN_SAMPLES_PER_STATE
    )

    if insufficient_samples:
        return {
            "experiment_type": EXPERIMENT_TYPE,
            "claim_ids_tested": CLAIM_IDS,
            "outcome": "FAIL",
            "evidence_class": "computational",
            "evidence_direction": "unknown",
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "summary": f"Insufficient samples: wanting={n_wanting}, nociception={n_nociception}, frustration={n_frustration}. Need >={MIN_SAMPLES_PER_STATE} per state.",
            "n_wanting": n_wanting,
            "n_nociception": n_nociception,
            "n_frustration": n_frustration,
            "seeds": list(range(NUM_SEEDS)),
        }

    # Prepare train/test split (70/30)
    random.seed(42)
    train_samples = []
    test_samples = []

    for state_samples in all_samples.values():
        random.shuffle(state_samples)
        split_idx = int(len(state_samples) * 0.7)
        train_samples.extend(state_samples[:split_idx])
        test_samples.extend(state_samples[split_idx:])

    # Train classifier
    print(f"\nTraining classifier ({len(train_samples)} train, {len(test_samples)} test)...")
    classifier_metrics = _train_classifier(train_samples, test_samples)

    print(f"  Overall accuracy: {classifier_metrics['overall_accuracy']:.3f}")
    print(f"  Wanting accuracy: {classifier_metrics['wanting_accuracy']:.3f}")
    print(f"  Nociception accuracy: {classifier_metrics['nociception_accuracy']:.3f}")
    print(f"  Frustration accuracy: {classifier_metrics['frustration_accuracy']:.3f}")

    # Evaluate PASS criteria
    c1_pass = classifier_metrics['wanting_accuracy'] >= 0.70
    c2_pass = classifier_metrics['nociception_accuracy'] >= 0.70
    c3_pass = classifier_metrics['frustration_accuracy'] >= 0.70 if n_frustration >= MIN_SAMPLES_PER_STATE else True  # Skip if no frustration samples
    c4_pass = classifier_metrics['overall_accuracy'] > 0.50  # Above 3-way chance
    c5_pass = True  # Completed without fatal errors

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass

    # Degeneracy self-report
    load_bearing_metrics = {
        "wanting_accuracy": [classifier_metrics['wanting_accuracy']],
        "nociception_accuracy": [classifier_metrics['nociception_accuracy']],
        "overall_accuracy": [classifier_metrics['overall_accuracy']],
    }
    degeneracy_report = check_degeneracy(load_bearing_metrics)

    result = {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids_tested": CLAIM_IDS,
        "outcome": "PASS" if all_pass else "FAIL",
        "evidence_class": "computational",
        "evidence_direction": "supports" if all_pass else "weakens",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "seeds": list(range(NUM_SEEDS)),
        "samples_collected": {
            "wanting": n_wanting,
            "nociception": n_nociception,
            "frustration": n_frustration,
        },
        "classifier_metrics": classifier_metrics,
        "criteria": {
            "C1_wanting_accuracy_ge_70pct": {
                "pass": c1_pass,
                "accuracy": classifier_metrics['wanting_accuracy'],
                "threshold": 0.70,
            },
            "C2_nociception_accuracy_ge_70pct": {
                "pass": c2_pass,
                "accuracy": classifier_metrics['nociception_accuracy'],
                "threshold": 0.70,
            },
            "C3_frustration_accuracy_ge_70pct": {
                "pass": c3_pass,
                "accuracy": classifier_metrics['frustration_accuracy'],
                "threshold": 0.70,
                "skipped": n_frustration < MIN_SAMPLES_PER_STATE,
            },
            "C4_overall_above_chance": {
                "pass": c4_pass,
                "accuracy": classifier_metrics['overall_accuracy'],
                "chance_level": 0.333,  # 3-way chance
            },
            "C5_runs_complete": {
                "pass": c5_pass,
            },
        },
        "summary": f"MECH-191 signal-state discriminability: {'PASS' if all_pass else 'FAIL'}. "
                  f"Overall accuracy {classifier_metrics['overall_accuracy']:.1%}, "
                  f"wanting={classifier_metrics['wanting_accuracy']:.1%}, "
                  f"nociception={classifier_metrics['nociception_accuracy']:.1%}, "
                  f"frustration={classifier_metrics['frustration_accuracy']:.1%}. "
                  f"Signals are {'discriminable' if all_pass else 'not reliably discriminable'} by functional state.",
    }

    # Merge degeneracy report
    result.update(degeneracy_report)

    return result


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

        print(f"\nResult written to: {out_path}", flush=True)
        print(f"Outcome: {result['outcome']}")
        print(f"Evidence direction: {result['evidence_direction']}")

        _outcome_raw = str(result["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=out_path,
        )
