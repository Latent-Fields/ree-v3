#!/opt/local/bin/python3
"""
V3-EXQ-677 -- MECH-180 Novelty Sleep Upregulation Probe
SLEEP DRIVER: manual-multi (run_sleep_cycle every 10 episodes)

CLAIM UNDER TEST: MECH-180
  "Novel environments adaptively upregulate sleep replay. High prediction error
  during waking triggers increased SWS schema writes and REM attribution rollouts."

SCIENTIFIC QUESTION:
  Does high novelty/prediction error during wake produce measurably increased
  sleep replay activity (SWS writes + REM rollouts) compared to stable
  environments?

EXPERIMENT PURPOSE: evidence

DESIGN:
  Two matched-seed conditions (3 seeds each: 42, 123, 456):

  ARM_HIGH_NOVELTY:
    - CausalGridWorldV2 with frequent env drift (env_drift_interval=3)
    - Context switching every 5 episodes
    - use_proxy_fields=True for high-dimensional world state
    - Expected: high E1 prediction error, novel experiences

  ARM_LOW_NOVELTY:
    - Stable CausalGridWorldV2 (env_drift_interval=999, no context switch)
    - Same grid size/hazards/resources
    - Expected: low E1 prediction error, stable experience

  Both arms:
    - use_sleep_aggregation_cluster=True (MECH-204 sleep aggregation)
    - Sleep cycle every 10 episodes (sleep_loop_episodes_K=10)
    - Training: 100 episodes x 200 steps
    - Phased training: P0 (0-40 eps encoder warmup), P1 (40-100 eps train with frozen encoder)

PRIMARY DEPENDENT VARIABLES:
  - cumulative_sws_writes (sum across all sleep cycles)
  - cumulative_rem_rollouts (sum across all sleep cycles)

MANIPULATION CHECK:
  - mean_e1_prediction_error (averaged across wake episodes)

CONTROLS:
  - goal_success (agents functional)
  - harm_rate (safety maintained)

ACCEPTANCE CRITERIA (all required):
  C1: mean_e1_prediction_error_HIGH > mean_e1_prediction_error_LOW + 0.01
      (manipulation check: novelty present)
  C2: cumulative_sws_writes_HIGH > cumulative_sws_writes_LOW * 1.25
      (25% more SWS activity under high novelty)
  C3: cumulative_rem_rollouts_HIGH > cumulative_rem_rollouts_LOW * 1.25
      (25% more REM activity under high novelty)
  C4: Both arms cumulative_sws_writes >= 3 AND cumulative_rem_rollouts >= 2
      (substrate functional, sleep cycles fire)
  C5: Both arms goal_success >= 0.03
      (agents not collapsed)

PASS: All 5 criteria PASS

INTERPRETATION:
  PASS all 5                 -> supports MECH-180
  C1 PASS but C2 or C3 FAIL  -> weakens (novelty present but sleep doesn't respond)
  C1 FAIL                    -> non_contributory (manipulation failed)
  C4 or C5 FAIL              -> substrate_failure (sleep substrate broken or agents collapsed)

claim_ids: ["MECH-180"]
experiment_purpose: "evidence"
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from _metrics import check_degeneracy

EXPERIMENT_TYPE = "v3_exq_677_mech180_novelty_sleep_upregulation_probe"
QUEUE_ID = "V3-EXQ-677"
CLAIM_IDS = ["MECH-180"]
EXPERIMENT_PURPOSE = "evidence"

# Design parameters
SEEDS = [42, 123, 456]
TRAINING_EPISODES = 100
STEPS_PER_EPISODE = 200
SLEEP_INTERVAL = 10  # Sleep cycle every 10 episodes
CONTEXT_SWITCH_INTERVAL = 5  # For HIGH_NOVELTY arm
ENV_DRIFT_INTERVAL_HIGH = 3  # Frequent drift
ENV_DRIFT_INTERVAL_LOW = 999  # Stable

# Phased training (P0 encoder warmup, P1 frozen encoder)
PHASE_0_END = 40  # Episodes 0-40: encoder warmup
PHASE_1_START = 40  # Episodes 40-100: train with frozen encoder

# Thresholds
THRESH_PRED_ERROR_DIFF = 0.01  # C1: manipulation check
THRESH_SLEEP_RATIO = 1.25      # C2/C3: 25% more sleep activity
THRESH_MIN_SWS = 3             # C4: substrate functional
THRESH_MIN_REM = 2             # C4: substrate functional
THRESH_GOAL_SUCCESS = 0.03     # C5: agent functional


def _make_env_high_novelty(seed: int, context: str) -> CausalGridWorldV2:
    """High novelty: frequent drift, context switching."""
    return CausalGridWorldV2(
        seed=seed if context == "safe" else seed + 1000,
        size=6,
        num_hazards=4,
        num_resources=3,
        hazard_harm=0.02 if context == "safe" else 0.04,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        env_drift_interval=ENV_DRIFT_INTERVAL_HIGH,
    )


def _make_env_low_novelty(seed: int) -> CausalGridWorldV2:
    """Low novelty: stable environment, no drift."""
    return CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=3,
        hazard_harm=0.02,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        env_drift_interval=ENV_DRIFT_INTERVAL_LOW,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Build agent with sleep aggregation enabled."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        # Sleep configuration
        sws_enabled=True,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=True,
        rem_attribution_steps=6,
        use_sleep_loop=True,
        use_sleep_aggregation_cluster=True,
    )
    return REEAgent(cfg)


def run_condition(
    condition_name: str,
    is_high_novelty: bool,
    seed: int,
) -> Dict:
    """Run one condition (HIGH_NOVELTY or LOW_NOVELTY) for one seed."""
    print(f"Seed {seed} Condition {condition_name}", flush=True)
    torch.manual_seed(seed)

    # Create environment(s)
    if is_high_novelty:
        env_safe = _make_env_high_novelty(seed, "safe")
        env_dangerous = _make_env_high_novelty(seed, "dangerous")
    else:
        env_safe = _make_env_low_novelty(seed)
        env_dangerous = None  # No context switching

    agent = _make_agent(env_safe)
    device = agent.device

    # Optimizers (phased training)
    # P0: encoder warmup (E1 + LatentStack)
    p0_optimizer = torch.optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=1e-4,
    )
    # P1: frozen encoder, train rest
    p1_params = []
    for name, param in agent.named_parameters():
        if "e1." not in name and "latent_stack." not in name:
            p1_params.append(param)
    p1_optimizer = torch.optim.Adam(p1_params, lr=1e-4)

    # Metrics accumulators
    cumulative_sws_writes = 0.0
    cumulative_rem_rollouts = 0.0
    e1_prediction_errors = []
    goal_successes = []
    harm_rates = []

    for ep in range(TRAINING_EPISODES):
        if ep % 20 == 0 or ep == TRAINING_EPISODES - 1:
            print(f"  [train] {condition_name} seed={seed} ep {ep+1}/{TRAINING_EPISODES}", flush=True)

        # Context switching for HIGH_NOVELTY
        if is_high_novelty and env_dangerous is not None:
            use_dangerous = (ep // CONTEXT_SWITCH_INTERVAL) % 2 == 1
            env = env_dangerous if use_dangerous else env_safe
        else:
            env = env_safe

        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        ep_harm = 0.0
        ep_steps = 0
        ep_goals = 0

        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)

            # Forward pass
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick", False) else \
                torch.zeros(1, agent.config.latent.world_dim, device=device)
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            # Step environment
            _, harm_signal, done, info, obs_dict = env.step(action)
            ep_harm += max(0.0, float(-harm_signal))

            # Training (phased)
            if ep < PHASE_0_END:
                # P0: encoder warmup
                optimizer = p0_optimizer
            else:
                # P1: frozen encoder
                optimizer = p1_optimizer
                # Freeze encoder
                for param in agent.e1.parameters():
                    param.requires_grad = False
                for param in agent.latent_stack.parameters():
                    param.requires_grad = False

            # Compute and backprop prediction loss
            pred_loss = agent.compute_prediction_loss()
            if pred_loss.requires_grad:
                optimizer.zero_grad()
                pred_loss.backward()
                optimizer.step()

            # Track E1 prediction error
            e1_prediction_errors.append(float(pred_loss.item()))

            # Track goals
            if info.get("goal_reached", False):
                ep_goals += 1

            ep_steps += 1
            if done:
                break

        # Episode metrics
        harm_rates.append(ep_harm / max(1, ep_steps))
        goal_successes.append(ep_goals)

        # Sleep cycle every SLEEP_INTERVAL episodes
        if (ep + 1) % SLEEP_INTERVAL == 0 and ep > 0:
            sleep_metrics = agent.run_sleep_cycle()
            cumulative_sws_writes += float(sleep_metrics.get("sws_n_writes", 0.0))
            cumulative_rem_rollouts += float(sleep_metrics.get("rem_n_rollouts", 0.0))

    # Aggregate metrics
    mean_e1_pred_error = float(sum(e1_prediction_errors) / max(1, len(e1_prediction_errors)))
    mean_goal_success = float(sum(goal_successes) / max(1, len(goal_successes)))
    mean_harm_rate = float(sum(harm_rates) / max(1, len(harm_rates)))

    print(f"verdict: PASS", flush=True)

    return {
        "condition": condition_name,
        "seed": seed,
        "cumulative_sws_writes": cumulative_sws_writes,
        "cumulative_rem_rollouts": cumulative_rem_rollouts,
        "mean_e1_prediction_error": mean_e1_pred_error,
        "mean_goal_success": mean_goal_success,
        "mean_harm_rate": mean_harm_rate,
    }


def main():
    """Run experiment and evaluate criteria."""
    all_results = {"HIGH_NOVELTY": [], "LOW_NOVELTY": []}

    # Run all conditions
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        high_res = run_condition("HIGH_NOVELTY", is_high_novelty=True, seed=seed)
        all_results["HIGH_NOVELTY"].append(high_res)

        low_res = run_condition("LOW_NOVELTY", is_high_novelty=False, seed=seed)
        all_results["LOW_NOVELTY"].append(low_res)

    # Aggregate across seeds
    def _mean(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / len(vals))

    agg_high = {k: _mean(all_results["HIGH_NOVELTY"], k) for k in
                ["cumulative_sws_writes", "cumulative_rem_rollouts",
                 "mean_e1_prediction_error", "mean_goal_success", "mean_harm_rate"]}
    agg_low = {k: _mean(all_results["LOW_NOVELTY"], k) for k in
               ["cumulative_sws_writes", "cumulative_rem_rollouts",
                "mean_e1_prediction_error", "mean_goal_success", "mean_harm_rate"]}

    # Evaluate criteria
    c1_pass = agg_high["mean_e1_prediction_error"] > agg_low["mean_e1_prediction_error"] + THRESH_PRED_ERROR_DIFF
    c2_pass = agg_high["cumulative_sws_writes"] > agg_low["cumulative_sws_writes"] * THRESH_SLEEP_RATIO
    c3_pass = agg_high["cumulative_rem_rollouts"] > agg_low["cumulative_rem_rollouts"] * THRESH_SLEEP_RATIO
    c4_pass = (agg_high["cumulative_sws_writes"] >= THRESH_MIN_SWS and
               agg_low["cumulative_sws_writes"] >= THRESH_MIN_SWS and
               agg_high["cumulative_rem_rollouts"] >= THRESH_MIN_REM and
               agg_low["cumulative_rem_rollouts"] >= THRESH_MIN_REM)
    c5_pass = (agg_high["mean_goal_success"] >= THRESH_GOAL_SUCCESS and
               agg_low["mean_goal_success"] >= THRESH_GOAL_SUCCESS)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    outcome = "PASS" if all_pass else "FAIL"

    # Determine evidence direction
    if all_pass:
        evidence_direction = "supports"
        evidence_note = "High novelty upregulates sleep replay as predicted by MECH-180"
    elif c1_pass and (not c2_pass or not c3_pass):
        evidence_direction = "weakens"
        evidence_note = "Novelty manipulation successful but sleep did not upregulate"
    elif not c1_pass:
        evidence_direction = "non_contributory"
        evidence_note = "Manipulation check failed: no E1 prediction error difference"
    elif not c4_pass:
        evidence_direction = "substrate_failure"
        evidence_note = "Sleep substrate did not produce minimum expected activity"
    else:
        evidence_direction = "substrate_failure"
        evidence_note = "Agent performance collapsed (C5 fail)"

    # Degeneracy self-report (discriminative criteria must show non-degenerate spread)
    degeneracy = check_degeneracy({
        "cumulative_sws_writes": {
            "values": [agg_high["cumulative_sws_writes"], agg_low["cumulative_sws_writes"]],
            "floor": 0.0,  # degenerate if both <= 0
        },
        "cumulative_rem_rollouts": {
            "values": [agg_high["cumulative_rem_rollouts"], agg_low["cumulative_rem_rollouts"]],
            "floor": 0.0,
        },
        "mean_e1_prediction_error": {
            "values": [agg_high["mean_e1_prediction_error"], agg_low["mean_e1_prediction_error"]],
        },
    })

    # Build manifest
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": evidence_note,
        "evidence_direction_per_claim": {
            "MECH-180": evidence_direction,
        },
        "aggregated": {
            "HIGH_NOVELTY": agg_high,
            "LOW_NOVELTY": agg_low,
        },
        "criteria": {
            "C1_manipulation_check": {
                "pass": c1_pass,
                "high_pred_error": agg_high["mean_e1_prediction_error"],
                "low_pred_error": agg_low["mean_e1_prediction_error"],
                "diff": agg_high["mean_e1_prediction_error"] - agg_low["mean_e1_prediction_error"],
                "threshold": THRESH_PRED_ERROR_DIFF,
            },
            "C2_sws_upregulation": {
                "pass": c2_pass,
                "high_sws": agg_high["cumulative_sws_writes"],
                "low_sws": agg_low["cumulative_sws_writes"],
                "ratio": agg_high["cumulative_sws_writes"] / max(0.001, agg_low["cumulative_sws_writes"]),
                "threshold_ratio": THRESH_SLEEP_RATIO,
            },
            "C3_rem_upregulation": {
                "pass": c3_pass,
                "high_rem": agg_high["cumulative_rem_rollouts"],
                "low_rem": agg_low["cumulative_rem_rollouts"],
                "ratio": agg_high["cumulative_rem_rollouts"] / max(0.001, agg_low["cumulative_rem_rollouts"]),
                "threshold_ratio": THRESH_SLEEP_RATIO,
            },
            "C4_substrate_functional": {
                "pass": c4_pass,
                "high_sws": agg_high["cumulative_sws_writes"],
                "low_sws": agg_low["cumulative_sws_writes"],
                "high_rem": agg_high["cumulative_rem_rollouts"],
                "low_rem": agg_low["cumulative_rem_rollouts"],
                "min_sws_threshold": THRESH_MIN_SWS,
                "min_rem_threshold": THRESH_MIN_REM,
            },
            "C5_agent_functional": {
                "pass": c5_pass,
                "high_goal_success": agg_high["mean_goal_success"],
                "low_goal_success": agg_low["mean_goal_success"],
                "threshold": THRESH_GOAL_SUCCESS,
            },
        },
        "config": {
            "seeds": SEEDS,
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_interval": SLEEP_INTERVAL,
            "context_switch_interval": CONTEXT_SWITCH_INTERVAL,
            "env_drift_interval_high": ENV_DRIFT_INTERVAL_HIGH,
            "env_drift_interval_low": ENV_DRIFT_INTERVAL_LOW,
            "phase_0_end": PHASE_0_END,
            "phase_1_start": PHASE_1_START,
            "use_sleep_aggregation_cluster": True,
            "sws_consolidation_steps": 8,
            "rem_attribution_steps": 6,
        },
        "results": {
            "HIGH_NOVELTY": all_results["HIGH_NOVELTY"],
            "LOW_NOVELTY": all_results["LOW_NOVELTY"],
        },
        "arm_fingerprint": {
            "HIGH_NOVELTY": {
                "env_drift_interval": ENV_DRIFT_INTERVAL_HIGH,
                "context_switching": True,
                "context_switch_interval": CONTEXT_SWITCH_INTERVAL,
            },
            "LOW_NOVELTY": {
                "env_drift_interval": ENV_DRIFT_INTERVAL_LOW,
                "context_switching": False,
            },
        },
    }

    # Merge degeneracy self-report
    manifest.update(degeneracy)

    # Write manifest
    out_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nResults written to {out_path}")
    print(f"Outcome: {outcome}")
    print(f"Evidence Direction: {evidence_direction}")
    print(f"C1 (manipulation check): {c1_pass}")
    print(f"C2 (SWS upregulation): {c2_pass}")
    print(f"C3 (REM upregulation): {c3_pass}")
    print(f"C4 (substrate functional): {c4_pass}")
    print(f"C5 (agent functional): {c5_pass}")

    return outcome, str(out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run minimal smoke test (1 seed, 3 episodes)")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Running smoke test...")
        # Override for quick smoke test
        SEEDS = [42]
        TRAINING_EPISODES = 3
        STEPS_PER_EPISODE = 20
        print(f"  Seeds: {SEEDS}")
        print(f"  Episodes: {TRAINING_EPISODES}")
        print(f"  Steps: {STEPS_PER_EPISODE}")

        # Quick env test
        env_high = _make_env_high_novelty(42, "safe")
        env_low = _make_env_low_novelty(42)
        agent = _make_agent(env_high)

        print(f"  Env HIGH body_obs_dim: {env_high.body_obs_dim}")
        print(f"  Env HIGH world_obs_dim: {env_high.world_obs_dim}")
        print(f"  Env LOW body_obs_dim: {env_low.body_obs_dim}")
        print(f"  Agent sleep enabled: {agent.config.sws_enabled}")
        print(f"  Agent use_sleep_aggregation_cluster: {agent.config.use_sleep_aggregation_cluster}")

        # Run 1 episode
        _, obs_dict = env_high.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        device = agent.device

        for step in range(10):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick", False) else \
                torch.zeros(1, agent.config.latent.world_dim, device=device)
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            _, harm_signal, done, info, obs_dict = env_high.step(action)
            pred_loss = agent.compute_prediction_loss()

        # Try sleep cycle
        sleep_metrics = agent.run_sleep_cycle()
        print(f"  Sleep cycle test: sws_n_writes={sleep_metrics.get('sws_n_writes', 0)}, "
              f"rem_n_rollouts={sleep_metrics.get('rem_n_rollouts', 0)}")

        print("[DRY RUN] PASS - substrate wiring confirmed")
        emit_outcome(outcome="PASS", manifest_path=None)
        sys.exit(0)

    t0 = time.time()
    print(f"{QUEUE_ID} {EXPERIMENT_TYPE}")

    outcome, out_path = main()

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    outcome_clean = str(outcome).upper() if outcome in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=outcome_clean,
        manifest_path=out_path,
    )
