"""
V3-EXQ-672 -- MECH-057b: Hippocampal Trajectory Completion Gate

Claims: MECH-057b

Motivation (2026-06-11):
  MECH-057b: "Hippocampal sequence completion must be verified before candidates
  are eligible for E3 selection."

  Split from MECH-057 (2026-03-15). MECH-057a tested the action-loop gate (BetaGate
  blocking E3->action propagation during committed sequences). MECH-057b tests the
  thought-loop gate: trajectory candidates from HippocampalModule should be
  filtered for completion quality before reaching E3.

  Current V3 substrate: HippocampalModule.propose_trajectories() generates
  candidates via CEM, scoring on residue cost. All candidates reach E3. The
  completion_signal is computed but only feeds BetaGate (MECH-057a), not a
  trajectory promotion gate.

  Experimental design:
    ARM_0_NO_GATE: baseline - all candidates reach E3 (current behavior)
    ARM_1_COMPLETION_GATE: filter candidates by completion quality before E3
      - Compute residue score for each trajectory
      - Suppress high-residue (incomplete) trajectories from E3 input
      - Only promote candidates with residue < threshold
      - Minimum candidates: always pass >=2 best candidates to avoid deadlock

  Key metrics:
    mean_harm_rate: fraction of steps with harm < 0 (primary outcome)
    mean_completion_signal: hippocampal completion quality (secondary)
    mean_e3_candidate_count: how many candidates E3 evaluates
    mean_filtered_fraction: fraction of candidates filtered by gate (ARM_1 only)
    mean_precision: E3 commitment precision
    goal_success_rate: fraction of episodes reaching goal

  PASS criteria (ALL must hold):
    C1: mean_harm_rate_ARM_1 < mean_harm_rate_ARM_0 * 0.90 (10% reduction)
    C2: mean_completion_signal_ARM_1 > mean_completion_signal_ARM_0 * 1.05
    C3: mean_filtered_fraction_ARM_1 >= 0.15 (gate actually filters)
    C4: mean_e3_candidate_count_ARM_1 >= 2.0 (not deadlocked)
    C5: mean_precision_ARM_1 >= 0.5 (E3 functional)
    C6: goal_success_rate_ARM_0 >= 0.05 (baseline not collapsed)
    C7: No fatal errors

  Non-vacuity gates:
    - Baseline must achieve some goal success (C6)
    - Gate must actually filter candidates (C3)
    - E3 must remain functional with filtered input (C4, C5)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.predictors.e2_fast import Trajectory
from experiment_protocol import emit_outcome


EXPERIMENT_TYPE = "v3_exq_672_mech057b_trajectory_promotion_gate"
CLAIM_IDS = ["MECH-057b"]


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _filter_trajectories_by_completion(
    agent: REEAgent,
    trajectories: List[Trajectory],
    completion_threshold: float = 2.0,
    min_candidates: int = 2,
) -> Tuple[List[Trajectory], float]:
    """
    Filter trajectories by completion quality (residue cost).

    Returns:
        (filtered_trajectories, filtered_fraction)

    Filtering rule:
        - Score each trajectory by residue cost (same as HippocampalModule._score_trajectory)
        - Suppress trajectories with score > completion_threshold
        - Always pass at least min_candidates (best-scoring) to avoid deadlock
    """
    if len(trajectories) <= min_candidates:
        return trajectories, 0.0

    # Score trajectories (lower is better)
    scores = []
    for traj in trajectories:
        with torch.no_grad():
            score = agent.hippocampal._score_trajectory(traj)
            scores.append(float(score.item() if isinstance(score, torch.Tensor) else score))

    # Sort by score (ascending)
    sorted_indices = sorted(range(len(trajectories)), key=lambda i: scores[i])

    # Filter: keep trajectories with score <= threshold, or top min_candidates
    kept_indices = []
    for idx in sorted_indices:
        if scores[idx] <= completion_threshold or len(kept_indices) < min_candidates:
            kept_indices.append(idx)

    filtered_trajectories = [trajectories[i] for i in kept_indices]
    filtered_fraction = 1.0 - (len(filtered_trajectories) / len(trajectories))

    return filtered_trajectories, filtered_fraction


def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    use_completion_gate: bool = False,
) -> Dict:
    """Standard training loop with optional trajectory completion gate."""
    agent.train()
    total_harm = 0
    total_steps = 0
    e3_tick_count = 0
    filtered_fractions: List[float] = []
    e3_candidate_counts: List[int] = []
    completion_signals: List[float] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_count += 1

                # Apply completion gate if enabled
                if use_completion_gate:
                    candidates, filtered_frac = _filter_trajectories_by_completion(
                        agent, candidates, completion_threshold=2.0, min_candidates=2
                    )
                    filtered_fractions.append(filtered_frac)
                else:
                    filtered_fractions.append(0.0)

                e3_candidate_counts.append(len(candidates))

                # Compute completion signal (diagnostic)
                comp_signal = agent.hippocampal.compute_completion_signal(candidates)
                completion_signals.append(comp_signal)

                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action
            else:
                action = agent._last_action
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            total_steps += 1
            if harm_signal < 0:
                total_harm += 1

            # E1 training
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            harm_rate = total_harm / max(1, total_steps)
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm_rate={harm_rate:.4f}"
                f"  e3_ticks={e3_tick_count}"
                f"  mean_candidates={_mean_safe(e3_candidate_counts):.2f}",
                flush=True,
            )

    return {
        "total_harm": total_harm,
        "total_steps": total_steps,
        "e3_tick_count": e3_tick_count,
        "filtered_fractions": filtered_fractions,
        "e3_candidate_counts": e3_candidate_counts,
        "completion_signals": completion_signals,
    }


def _eval(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    use_completion_gate: bool = False,
) -> Dict:
    """Eval loop measuring harm rate, completion signal, and goal success."""
    agent.eval()
    total_harm = 0
    total_steps = 0
    goal_successes = 0
    e3_tick_count = 0
    filtered_fractions: List[float] = []
    e3_candidate_counts: List[int] = []
    completion_signals: List[float] = []
    precisions: List[float] = []

    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        reached_goal = False

        for step_idx in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            try:
                if ticks.get("e3_tick", False) and candidates:
                    e3_tick_count += 1

                    # Apply completion gate if enabled
                    if use_completion_gate:
                        candidates, filtered_frac = _filter_trajectories_by_completion(
                            agent, candidates, completion_threshold=2.0, min_candidates=2
                        )
                        filtered_fractions.append(filtered_frac)
                    else:
                        filtered_fractions.append(0.0)

                    e3_candidate_counts.append(len(candidates))

                    with torch.no_grad():
                        comp_signal = agent.hippocampal.compute_completion_signal(candidates)
                        completion_signals.append(comp_signal)

                        result = agent.e3.select(candidates, temperature=1.0)
                        action = result.selected_action.detach()
                        agent._last_action = action

                        precisions.append(float(result.precision))
                else:
                    action = agent._last_action
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                total_steps += 1
                if harm_signal < 0:
                    total_harm += 1

                # Check goal reach (resource collected)
                if not reached_goal and info.get("resource_collected", False):
                    reached_goal = True
                    goal_successes += 1

                z_self_prev = latent.z_self.detach()
                action_prev = action.detach()
                if done:
                    break

            except Exception as e:
                print(f"  [eval] ERROR: {e}", flush=True)
                break

    harm_rate = total_harm / max(1, total_steps)
    goal_success_rate = goal_successes / max(1, num_episodes)
    mean_completion_signal = _mean_safe(completion_signals)
    mean_e3_candidate_count = _mean_safe(e3_candidate_counts)
    mean_filtered_fraction = _mean_safe(filtered_fractions)
    mean_precision = _mean_safe(precisions)

    return {
        "mean_harm_rate": harm_rate,
        "goal_success_rate": goal_success_rate,
        "mean_completion_signal": mean_completion_signal,
        "mean_e3_candidate_count": mean_e3_candidate_count,
        "mean_filtered_fraction": mean_filtered_fraction,
        "mean_precision": mean_precision,
        "e3_tick_count": e3_tick_count,
    }


def _run_one_seed(seed: int, arm: str, use_completion_gate: bool, dry_run: bool) -> Dict:
    """Run one seed of the experiment."""
    print(f"[{arm}] Seed {seed} starting...", flush=True)

    random.seed(seed)
    torch.manual_seed(seed)

    # Environment with resources as goals
    env = CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=2,
        num_resources=1,
        hazard_harm=0.05,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.03,
        proximity_benefit_scale=0.02,
        resource_benefit=0.3,
    )

    # Config from environment dimensions
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=5,
    )
    config.hidden_dim = 128
    config.horizon = 4
    config.num_candidates = 8
    config.use_goal = True
    config.goal_dim = 32

    # Agent
    agent = REEAgent(config)
    optimizer = optim.Adam(agent.e1.parameters(), lr=1e-4)

    # Training
    num_train_episodes = 2 if dry_run else 500
    steps_per_episode = 20 if dry_run else 200

    train_result = _train(
        agent,
        env,
        optimizer,
        num_train_episodes,
        steps_per_episode,
        config.e1.world_dim,
        use_completion_gate=use_completion_gate,
    )

    # Eval
    num_eval_episodes = 2 if dry_run else 100
    eval_result = _eval(
        agent,
        env,
        num_eval_episodes,
        steps_per_episode,
        config.e1.world_dim,
        use_completion_gate=use_completion_gate,
    )

    print(
        f"[{arm}] Seed {seed} complete: "
        f"harm_rate={eval_result['mean_harm_rate']:.4f} "
        f"goal_success={eval_result['goal_success_rate']:.4f} "
        f"completion={eval_result['mean_completion_signal']:.4f} "
        f"candidates={eval_result['mean_e3_candidate_count']:.2f} "
        f"filtered={eval_result['mean_filtered_fraction']:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "arm": arm,
        "use_completion_gate": use_completion_gate,
        "train_result": train_result,
        "eval_result": eval_result,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"V3-EXQ-672: MECH-057b Trajectory Promotion Gate", flush=True)
    print(f"Dry run: {args.dry_run}", flush=True)

    seeds = [42] if args.dry_run else [42, 43, 44]
    results = []
    fatal = 0

    # ARM_0: No gate (baseline)
    print("\n=== ARM_0_NO_GATE ===", flush=True)
    for seed in seeds:
        try:
            result = _run_one_seed(seed, "ARM_0_NO_GATE", use_completion_gate=False, dry_run=args.dry_run)
            results.append(result)
        except Exception as e:
            print(f"[ARM_0_NO_GATE] Seed {seed} FATAL: {e}", flush=True)
            fatal += 1

    # ARM_1: Completion gate
    print("\n=== ARM_1_COMPLETION_GATE ===", flush=True)
    for seed in seeds:
        try:
            result = _run_one_seed(seed, "ARM_1_COMPLETION_GATE", use_completion_gate=True, dry_run=args.dry_run)
            results.append(result)
        except Exception as e:
            print(f"[ARM_1_COMPLETION_GATE] Seed {seed} FATAL: {e}", flush=True)
            fatal += 1

    # Aggregate results
    arm0_results = [r["eval_result"] for r in results if r["arm"] == "ARM_0_NO_GATE"]
    arm1_results = [r["eval_result"] for r in results if r["arm"] == "ARM_1_COMPLETION_GATE"]

    if len(arm0_results) == 0 or len(arm1_results) == 0:
        print("ERROR: Insufficient results to evaluate", flush=True)
        print(f"ARM_0 results: {len(arm0_results)}, ARM_1 results: {len(arm1_results)}", flush=True)
        print(f"Fatal errors: {fatal}", flush=True)
        # emit_outcome with FAIL on all criteria
        emit_outcome(
            "FAIL",
            manifest_path=None,
            extra={
                "error": "insufficient_results",
                "arm0_count": len(arm0_results),
                "arm1_count": len(arm1_results),
                "fatal_count": fatal,
                "criteria": {
                    "C1_harm_reduction": False,
                    "C2_completion_improvement": False,
                    "C3_gate_filters": False,
                    "C4_not_deadlocked": False,
                    "C5_e3_functional": False,
                    "C6_baseline_not_collapsed": False,
                    "C7_no_fatal_errors": False,
                },
            },
        )
        return

    mean_harm_rate_arm0 = _mean_safe([r["mean_harm_rate"] for r in arm0_results])
    mean_harm_rate_arm1 = _mean_safe([r["mean_harm_rate"] for r in arm1_results])
    mean_completion_signal_arm0 = _mean_safe([r["mean_completion_signal"] for r in arm0_results])
    mean_completion_signal_arm1 = _mean_safe([r["mean_completion_signal"] for r in arm1_results])
    mean_filtered_fraction_arm1 = _mean_safe([r["mean_filtered_fraction"] for r in arm1_results])
    mean_e3_candidate_count_arm1 = _mean_safe([r["mean_e3_candidate_count"] for r in arm1_results])
    mean_precision_arm1 = _mean_safe([r["mean_precision"] for r in arm1_results])
    goal_success_rate_arm0 = _mean_safe([r["goal_success_rate"] for r in arm0_results])

    print("\n=== AGGREGATE RESULTS ===", flush=True)
    print(f"ARM_0 harm_rate: {mean_harm_rate_arm0:.4f}", flush=True)
    print(f"ARM_1 harm_rate: {mean_harm_rate_arm1:.4f}", flush=True)
    print(f"ARM_0 completion: {mean_completion_signal_arm0:.4f}", flush=True)
    print(f"ARM_1 completion: {mean_completion_signal_arm1:.4f}", flush=True)
    print(f"ARM_1 filtered_fraction: {mean_filtered_fraction_arm1:.4f}", flush=True)
    print(f"ARM_1 e3_candidates: {mean_e3_candidate_count_arm1:.2f}", flush=True)
    print(f"ARM_1 precision: {mean_precision_arm1:.4f}", flush=True)
    print(f"ARM_0 goal_success: {goal_success_rate_arm0:.4f}", flush=True)

    # Check PASS criteria
    c1 = mean_harm_rate_arm1 < mean_harm_rate_arm0 * 0.90
    c2 = mean_completion_signal_arm1 > mean_completion_signal_arm0 * 1.05
    c3 = mean_filtered_fraction_arm1 >= 0.15
    c4 = mean_e3_candidate_count_arm1 >= 2.0
    c5 = mean_precision_arm1 >= 0.5
    c6 = goal_success_rate_arm0 >= 0.05
    c7 = fatal == 0

    criteria_met = {
        "C1_harm_reduction": c1,
        "C2_completion_improvement": c2,
        "C3_gate_filters": c3,
        "C4_not_deadlocked": c4,
        "C5_e3_functional": c5,
        "C6_baseline_not_collapsed": c6,
        "C7_no_fatal_errors": c7,
    }

    print("\n=== PASS CRITERIA ===", flush=True)
    for crit, met in criteria_met.items():
        print(f"{crit}: {'PASS' if met else 'FAIL'}", flush=True)

    outcome = "PASS" if all(criteria_met.values()) else "FAIL"

    # Emit outcome
    emit_outcome(
        outcome,
        manifest_path=None,
        extra={
            "mean_harm_rate_ARM_0": mean_harm_rate_arm0,
            "mean_harm_rate_ARM_1": mean_harm_rate_arm1,
            "mean_completion_signal_ARM_0": mean_completion_signal_arm0,
            "mean_completion_signal_ARM_1": mean_completion_signal_arm1,
            "mean_filtered_fraction_ARM_1": mean_filtered_fraction_arm1,
            "mean_e3_candidate_count_ARM_1": mean_e3_candidate_count_arm1,
            "mean_precision_ARM_1": mean_precision_arm1,
            "goal_success_rate_ARM_0": goal_success_rate_arm0,
            "criteria": criteria_met,
            "fatal_count": fatal,
        },
    )


if __name__ == "__main__":
    main()
