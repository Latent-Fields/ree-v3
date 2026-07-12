"""
V3-EXQ-672a -- MECH-057b: Hippocampal Trajectory Completion Gate

Claims: MECH-057b
Supersedes: V3-EXQ-672 (self-reported DEGENERATE, scoring-excluded 2026-06-13 AM
            governance). 672 set an ABSOLUTE completion gate
            (completion_threshold=2.0 on the residue-completion score). The
            residue terrain over this small/low-harm env produces scores near 0
            with little spread, so the absolute bar sat above the entire candidate
            distribution -> mean_filtered_fraction_ARM_1 = 0 (gate inert) -> the
            C1/C2 cross-arm contrast was vacuous. 672a replaces the absolute bar
            with a DATA-RELATIVE quantile bar that adapts to the residue
            magnitude, and adds a readiness precondition that self-routes
            substrate_not_ready_requeue when the gate still cannot bite.

Motivation (2026-06-11):
  MECH-057b: "Hippocampal sequence completion must be verified before candidates
  are eligible for E3 selection."

  Split from MECH-057 (2026-03-15). MECH-057a tested the action-loop gate (BetaGate
  blocking E3->action propagation during committed sequences). MECH-057b tests the
  thought-loop gate: trajectory candidates from HippocampalModule should be
  filtered for completion quality before reaching E3.

  Current V3 substrate: HippocampalModule.propose_trajectories() generates
  candidates via CEM, scoring on residue cost. All candidates reach E3. The
  completion_signal is computed (compute_completion_signal = sigmoid(-best_score*0.5)
  over the same _score_trajectory residue-completion score) but only feeds BetaGate
  (MECH-057a), not a trajectory promotion gate. This experiment installs that
  promotion gate at the experiment-harness candidate->E3 seam.

  Experimental design:
    ARM_0_NO_GATE: baseline - all candidates reach E3 (current behavior)
    ARM_1_COMPLETION_GATE: promote candidates by RELATIVE completion quality
      before E3.
      - Score each trajectory by _score_trajectory (residue-completion cost;
        lower = more complete; same score the substrate's completion_signal uses).
      - Compute a per-tick RELATIVE bar: the (1 - DROP_FRACTION) quantile of the
        current candidate pool's scores. A trajectory is PROMOTED only if its
        score is at or below that bar (completion at least as verified as the
        pool's drop-quantile).
      - Always promote at least MIN_CANDIDATES best candidates (avoid deadlock).
      - The bar is data-relative, so it adapts to the substrate's residue
        magnitude. CRUCIALLY: when the candidate pool has no score spread, the
        quantile equals every score -> nothing is filtered -> filtered_fraction
        collapses to ~0. So mean_filtered_fraction is a faithful readout of
        whether the gate could verify-and-suppress at all -- it is the readiness
        precondition statistic (see below).

  Key metrics:
    mean_harm_rate: fraction of steps with harm < 0 (primary outcome)
    mean_completion_signal: hippocampal completion quality (secondary)
    mean_e3_candidate_count: how many candidates E3 evaluates (post-gate, ARM_1)
    mean_filtered_fraction: fraction of candidates filtered by the gate (ARM_1)
    mean_candidate_score_spread: mean per-tick (max-min) of candidate scores
      (ARM_1) -- the non-vacuity readout; the quantile bar can only bite when
      this is > 0.
    mean_precision: E3 commitment precision
    goal_success_rate: fraction of episodes reaching goal

  Readiness precondition (P0-style, evaluated on the ARM_1 positive control):
    The load-bearing C1/C2 criteria read a CROSS-ARM difference that can only
    exist if ARM_1 actually filtered a non-trivial fraction. So BEFORE trusting a
    PASS/FAIL, the run asserts the gate engaged:
      mean_filtered_fraction_ARM_1 >= FILTER_FLOOR (the SAME statistic the
      verify-and-suppress mechanism produces; it folds in the score-spread
      requirement because the relative bar cannot filter a flat pool).
    If unmet -> interpretation.label = "substrate_not_ready_requeue",
    non_degenerate = False (scoring-excluded by the indexer), NOT a MECH-057b
    weakens. This is the explicit guard against the V3-EXQ-672 degeneracy
    recurring.

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
    - Gate must actually filter candidates (C3 + the readiness precondition)
    - E3 must remain functional with filtered input (C4, C5)
    - Cross-arm C1/C2 channels must move across seeds (check_degeneracy net)
"""

import sys
import os
import json
import random
from datetime import datetime
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
from _metrics import check_degeneracy
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_672a_mech057b_trajectory_promotion_gate"
CLAIM_IDS = ["MECH-057b"]

# --- Pre-registered gate + readiness constants (defined in-script, not post-hoc) ---
# DROP_FRACTION: target fraction of the candidate pool the relative completion bar
#   suppresses (the least-complete = highest-residue candidates) before E3.
DROP_FRACTION = 0.4
# MIN_CANDIDATES: floor on promoted candidates (deadlock guard).
MIN_CANDIDATES = 2
# FILTER_FLOOR: readiness threshold on mean_filtered_fraction_ARM_1. Below this the
#   gate did not bite (no candidate-score spread for the quantile bar to act on) ->
#   self-route substrate_not_ready_requeue. This is the explicit guard against the
#   V3-EXQ-672 degeneracy (mean_filtered_fraction_ARM_1 = 0) recurring.
FILTER_FLOOR = 0.10


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _filter_trajectories_by_completion(
    agent: REEAgent,
    trajectories: List[Trajectory],
    drop_fraction: float = DROP_FRACTION,
    min_candidates: int = MIN_CANDIDATES,
) -> Tuple[List[Trajectory], float, float]:
    """
    Promote trajectories by RELATIVE completion quality (residue-completion score).

    Returns:
        (promoted_trajectories, filtered_fraction, score_range)

    Filtering rule (data-relative; supersedes the V3-EXQ-672 absolute threshold):
        - Score each trajectory by _score_trajectory (residue-completion cost;
          lower = more complete; identical to the score the substrate's
          compute_completion_signal consumes).
        - Compute a per-tick RELATIVE bar = the (1 - drop_fraction) quantile of the
          current candidate pool's scores. PROMOTE a trajectory only if its score
          is at or below that bar (completion at least as verified as the pool's
          drop-quantile). This adapts to the substrate's residue magnitude -- the
          absolute bar=2.0 in V3-EXQ-672 sat above the entire distribution and
          filtered nothing.
        - Always promote at least min_candidates (best-scoring) to avoid deadlock.

    Non-vacuity by construction: when the candidate pool has no score spread the
    quantile equals every score, so every score is <= the bar -> nothing is
    suppressed -> filtered_fraction collapses to ~0. score_range (max-min of the
    raw scores) is returned so the caller can confirm the pool was non-degenerate;
    filtered_fraction is the readiness statistic the precondition routes on.
    """
    n = len(trajectories)
    if n <= min_candidates:
        return trajectories, 0.0, 0.0

    # Score trajectories (lower is more complete)
    scores: List[float] = []
    for traj in trajectories:
        with torch.no_grad():
            score = agent.hippocampal._score_trajectory(traj)
            scores.append(float(score.item() if isinstance(score, torch.Tensor) else score))

    score_range = float(max(scores) - min(scores))

    # Relative completion bar: the (1 - drop_fraction) quantile of pool scores.
    score_t = torch.tensor(scores, dtype=torch.float64)
    threshold = float(torch.quantile(score_t, 1.0 - drop_fraction).item())

    sorted_indices = sorted(range(n), key=lambda i: scores[i])  # best (lowest) first
    kept_indices: List[int] = []
    for idx in sorted_indices:
        if scores[idx] <= threshold or len(kept_indices) < min_candidates:
            kept_indices.append(idx)

    promoted = [trajectories[i] for i in kept_indices]
    filtered_fraction = 1.0 - (len(promoted) / n)

    return promoted, filtered_fraction, score_range


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
                    candidates, filtered_frac, _score_range = _filter_trajectories_by_completion(
                        agent, candidates, drop_fraction=DROP_FRACTION, min_candidates=MIN_CANDIDATES
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
    candidate_score_ranges: List[float] = []
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
                        candidates, filtered_frac, score_range = _filter_trajectories_by_completion(
                            agent, candidates, drop_fraction=DROP_FRACTION, min_candidates=MIN_CANDIDATES
                        )
                        filtered_fractions.append(filtered_frac)
                        candidate_score_ranges.append(score_range)
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
    mean_candidate_score_spread = _mean_safe(candidate_score_ranges)
    mean_precision = _mean_safe(precisions)

    return {
        "mean_harm_rate": harm_rate,
        "goal_success_rate": goal_success_rate,
        "mean_completion_signal": mean_completion_signal,
        "mean_e3_candidate_count": mean_e3_candidate_count,
        "mean_filtered_fraction": mean_filtered_fraction,
        "mean_candidate_score_spread": mean_candidate_score_spread,
        "mean_precision": mean_precision,
        "e3_tick_count": e3_tick_count,
    }


def _run_one_seed(seed: int, arm: str, use_completion_gate: bool, dry_run: bool) -> Dict:
    """Run one seed of the experiment."""
    # Progress boundary (runner resets episodes_in_run on this line)
    print(f"Seed {seed} Condition {arm}", flush=True)
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
    # Progress verdict (runner increments runs_done + records elapsed per run).
    # Per-run completion marker; the scientific PASS/FAIL is aggregate (see main()).
    print("verdict: PASS", flush=True)

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

    print(f"V3-EXQ-672a: MECH-057b Trajectory Promotion Gate (relative completion bar)", flush=True)
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
        # FAIL on all criteria; emitted by the __main__ block (runner contract).
        return {
            "outcome": "FAIL",
            "manifest_path": None,
            "extra": {
                "error": "insufficient_results",
                "arm0_count": len(arm0_results),
                "arm1_count": len(arm1_results),
                "fatal_count": fatal,
                "non_degenerate": False,
                "degeneracy_reason": "insufficient results to evaluate (ARM_0/ARM_1 cells missing) -- no cross-arm contrast",
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
        }

    mean_harm_rate_arm0 = _mean_safe([r["mean_harm_rate"] for r in arm0_results])
    mean_harm_rate_arm1 = _mean_safe([r["mean_harm_rate"] for r in arm1_results])
    mean_completion_signal_arm0 = _mean_safe([r["mean_completion_signal"] for r in arm0_results])
    mean_completion_signal_arm1 = _mean_safe([r["mean_completion_signal"] for r in arm1_results])
    mean_filtered_fraction_arm1 = _mean_safe([r["mean_filtered_fraction"] for r in arm1_results])
    mean_candidate_score_spread_arm1 = _mean_safe(
        [r.get("mean_candidate_score_spread", 0.0) for r in arm1_results]
    )
    mean_e3_candidate_count_arm1 = _mean_safe([r["mean_e3_candidate_count"] for r in arm1_results])
    mean_precision_arm1 = _mean_safe([r["mean_precision"] for r in arm1_results])
    goal_success_rate_arm0 = _mean_safe([r["goal_success_rate"] for r in arm0_results])

    print("\n=== AGGREGATE RESULTS ===", flush=True)
    print(f"ARM_0 harm_rate: {mean_harm_rate_arm0:.4f}", flush=True)
    print(f"ARM_1 harm_rate: {mean_harm_rate_arm1:.4f}", flush=True)
    print(f"ARM_0 completion: {mean_completion_signal_arm0:.4f}", flush=True)
    print(f"ARM_1 completion: {mean_completion_signal_arm1:.4f}", flush=True)
    print(f"ARM_1 filtered_fraction: {mean_filtered_fraction_arm1:.4f}", flush=True)
    print(f"ARM_1 candidate_score_spread: {mean_candidate_score_spread_arm1:.4f}", flush=True)
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

    # ---- Non-degeneracy self-report (failure_autopsy_batch9 Structural Pattern 1) ----
    # The load-bearing discriminative criteria C1 (cross-arm harm) and C2 (cross-arm
    # completion_signal) press MECH-057b. Both read a channel that does not move if the
    # completion gate has no behavioural effect (the hippocampal completion_signal is a
    # flat readout, or the gate filters nothing) -- the 670/671/673 "vacuous read on an
    # unwritten/untrained channel" family. Build per-seed matched (ARM_0, ARM_1) pairs so
    # cross-seed variance cannot mask a within-seed byte-identical contrast
    # (metric_groups rationale); a NON-VACUITY gate on filtered_fraction additionally
    # marks the run degenerate when the ARM_1 manipulation never engaged. A degenerate
    # run is scoring-excluded by the indexer (non_contributory), NOT a MECH-057b weakens.
    by_seed_arm0 = {r["seed"]: r["eval_result"] for r in results if r["arm"] == "ARM_0_NO_GATE"}
    by_seed_arm1 = {r["seed"]: r["eval_result"] for r in results if r["arm"] == "ARM_1_COMPLETION_GATE"}
    matched_seeds = sorted(set(by_seed_arm0) & set(by_seed_arm1))
    completion_groups = [
        [by_seed_arm0[s]["mean_completion_signal"], by_seed_arm1[s]["mean_completion_signal"]]
        for s in matched_seeds
    ]
    harm_groups = [
        [by_seed_arm0[s]["mean_harm_rate"], by_seed_arm1[s]["mean_harm_rate"]]
        for s in matched_seeds
    ]
    _degen = check_degeneracy({
        "C2_completion_signal_cross_arm": {"groups": completion_groups},
        "C1_harm_rate_cross_arm": {"groups": harm_groups},
    })

    # --- Readiness precondition (self-route substrate_not_ready_requeue) ---
    # The load-bearing C1/C2 criteria route on a CROSS-ARM difference that can only
    # exist if ARM_1 actually filtered a non-trivial fraction. mean_filtered_fraction
    # is the SAME statistic the verify-and-suppress mechanism produces (and it folds
    # in the score-spread requirement: the relative quantile bar cannot bite a flat
    # candidate pool). Below FILTER_FLOOR the gate is inert -> the V3-EXQ-672
    # degeneracy -> substrate_not_ready_requeue, NOT a MECH-057b weakens.
    gate_engaged = mean_filtered_fraction_arm1 >= FILTER_FLOOR
    preconditions = [
        {
            "name": "arm1_gate_filters_nontrivial_fraction",
            "description": (
                "ARM_1 completion gate must suppress >= FILTER_FLOOR of the candidate "
                "pool; the relative quantile bar can only bite when candidate "
                "residue-completion scores have spread (positive control: gate "
                "enabled, >2 candidates/tick on the ARM_1 eval pool)"
            ),
            "measured": mean_filtered_fraction_arm1,
            "threshold": FILTER_FLOOR,
            "control": "ARM_1 eval candidate pool (gate enabled)",
            "met": bool(gate_engaged),
        }
    ]

    non_degenerate = bool(_degen["non_degenerate"]) and gate_engaged
    degeneracy_reason = _degen["degeneracy_reason"]
    interpretation_label = "ready"
    if not gate_engaged:
        interpretation_label = "substrate_not_ready_requeue"
        if not degeneracy_reason:
            degeneracy_reason = (
                f"completion gate suppressed < FILTER_FLOOR of candidates "
                f"(mean_filtered_fraction_ARM_1={mean_filtered_fraction_arm1:.3g} < "
                f"{FILTER_FLOOR}; mean_candidate_score_spread_ARM_1="
                f"{mean_candidate_score_spread_arm1:.3g}) -- ARM_1 manipulation inert "
                "(no residue-completion score spread for the relative bar to act on); "
                "the C1/C2 cross-arm contrast is vacuous -- substrate_not_ready_requeue"
            )

    interpretation = {
        "label": interpretation_label,
        "preconditions": preconditions,
        "criteria_non_degenerate": {
            "C1_harm_reduction": bool(gate_engaged),
            "C2_completion_improvement": bool(gate_engaged),
            "C3_gate_filters": bool(gate_engaged),
        },
    }

    print("\n=== READINESS / NON-DEGENERACY ===", flush=True)
    print(
        f"gate_engaged: {gate_engaged} (filtered_fraction={mean_filtered_fraction_arm1:.4f} "
        f">= FILTER_FLOOR={FILTER_FLOOR})  label: {interpretation_label}",
        flush=True,
    )
    print(f"non_degenerate: {non_degenerate}  reason: {degeneracy_reason or '(none)'}", flush=True)

    evidence_direction = "supports" if outcome == "PASS" else "weakens"
    extra = {
        "mean_harm_rate_ARM_0": mean_harm_rate_arm0,
        "mean_harm_rate_ARM_1": mean_harm_rate_arm1,
        "mean_completion_signal_ARM_0": mean_completion_signal_arm0,
        "mean_completion_signal_ARM_1": mean_completion_signal_arm1,
        "mean_filtered_fraction_ARM_1": mean_filtered_fraction_arm1,
        "mean_candidate_score_spread_ARM_1": mean_candidate_score_spread_arm1,
        "mean_e3_candidate_count_ARM_1": mean_e3_candidate_count_arm1,
        "mean_precision_ARM_1": mean_precision_arm1,
        "goal_success_rate_ARM_0": goal_success_rate_arm0,
        "criteria": criteria_met,
        "fatal_count": fatal,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "interpretation": interpretation,
    }

    # Write a flat-JSON evidence manifest with root-level non_degenerate so the indexer's
    # scoring-exclusion net can read it. dry-runs write no manifest (manifest_path=None).
    manifest_path = None
    if not args.dry_run:
        run_id = f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
        evidence_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = evidence_dir / f"{run_id}.json"
        manifest = {
            "experiment_type": EXPERIMENT_TYPE,
            "run_id": run_id,
            "queue_id": os.environ.get("REE_QUEUE_ID", "V3-EXQ-672a"),
            "supersedes": "v3_exq_672_mech057b_trajectory_promotion_gate",
            "claim_ids": CLAIM_IDS,
            "claim_ids_tested": CLAIM_IDS,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "experiment_purpose": "evidence",
            "outcome": outcome,
            "evidence_direction": evidence_direction,
            "evidence_direction_per_claim": {"MECH-057b": evidence_direction},
            "non_degenerate": non_degenerate,
            "degeneracy_reason": degeneracy_reason,
            "degenerate_metrics": _degen["degenerate_metrics"],
            "interpretation": interpretation,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "criteria": criteria_met,
            "metrics": {
                "mean_harm_rate_ARM_0": mean_harm_rate_arm0,
                "mean_harm_rate_ARM_1": mean_harm_rate_arm1,
                "mean_completion_signal_ARM_0": mean_completion_signal_arm0,
                "mean_completion_signal_ARM_1": mean_completion_signal_arm1,
                "mean_filtered_fraction_ARM_1": mean_filtered_fraction_arm1,
                "mean_candidate_score_spread_ARM_1": mean_candidate_score_spread_arm1,
                "mean_e3_candidate_count_ARM_1": mean_e3_candidate_count_arm1,
                "mean_precision_ARM_1": mean_precision_arm1,
                "goal_success_rate_ARM_0": goal_success_rate_arm0,
                "fatal_count": fatal,
            },
            "params": {
                "seeds": seeds,
                "size": 8, "num_hazards": 2, "num_resources": 1,
                "drop_fraction": DROP_FRACTION,
                "min_candidates": MIN_CANDIDATES,
                "filter_floor": FILTER_FLOOR,
            },
        }
        manifest_path = write_flat_manifest(
            manifest,
            evidence_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=None,
            script_path=Path(__file__),
        )
        print(f"[V3-EXQ-672] Manifest written: {manifest_path}", flush=True)

    # Outcome emitted by the __main__ block (runner contract; keeps emit_outcome
    # lexically inside `if __name__ == "__main__":` for validate_experiments).
    return {
        "outcome": outcome,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "extra": extra,
    }


if __name__ == "__main__":
    _result = main()
    emit_outcome(
        _result["outcome"],
        manifest_path=_result["manifest_path"],
        extra=_result["extra"],
    )
