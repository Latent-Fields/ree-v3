#!/usr/bin/env python3
"""
EXQ-244b: MECH-165 reverse replay diversity scheduler validation (corrected).

Supersedes V3-EXQ-244a. EXQ-244a's driver called agent.reset() as a "final
flush" at the end of Phase 1, which clears agent.theta_buffer -- so its
hand-rolled SWS loop read theta_buffer.recent as None every cycle, and the
"if recent is not None" guard silently skipped replay in EVERY condition
(including FORWARD_REPLAY, hence FORWARD == NO_REPLAY byte-for-byte, and
reverse_replayed == 0 in every BALANCED_REPLAY seed). See
REE_assembly/evidence/planning/mech165_reverse_replay_nonfiring_diagnosis_2026-08-08.md
for the full diagnosis. The hippocampal mechanism itself was verified
functional in that diagnosis session; only the driver was broken.

Corrections in this iteration:
  1. No "final flush" agent.reset() between Phase 1 and consolidation --
     theta_buffer.recent and agent._current_latent retain Phase-1's last
     sensed content into the SWS window.
  2. Consolidation is driven through the PRODUCTION replay path: repeatedly
     call agent.clock.advance() (no new sensing -- pure "sleep" ticks) and
     invoke agent._do_replay(...) on every e3_quiescent tick, exactly as
     agent.act()/act_with_split_obs() would internally during waking
     quiescent cycles (MECH-092). Reverse-firing is read from the substrate
     observability counter agent.hippocampal._diverse_replay_mode_counts
     ["reverse"] (landed 2026-08-08, ree-v3 origin/main f534772), not a
     bespoke count on directly-returned trajectories -- _do_replay discards
     its return value, exactly as the production caller does.
  3. Non-degeneracy is asserted IN-RUN: the run FAILs loudly (not a silent
     marginal PASS) if reverse_replayed == 0 in every BALANCED_REPLAY seed,
     or if FORWARD_REPLAY is byte-identical to NO_REPLAY across all seeds --
     the two degenerate outcomes EXQ-244a silently produced.
  4. Consolidation volume doubled (10 quiescent fires instead of 5, ~100
     clock ticks instead of ~50) to give replay content more opportunity to
     move Phase-2 behavior, per the claim's precondition #3 (exploration-
     generated source material must be able to differentiate FORWARD from
     BALANCED, not just be present in the buffer).

Design (unchanged from EXQ-244a except as above -- EXP-0105 redesign):
  Phase 1 (exploration): warmup with epsilon=0.5 random exploration, generating
  diverse behavioral trajectories into exploration buffer via
  agent._record_exploration_action(). Measure baseline action_entropy. All
  conditions run identically in Phase 1.

  Consolidation (offline, no env stepping): three conditions differ ONLY in
  replay mode, driven through agent._do_replay() on real e3_quiescent ticks:
    A) NO_REPLAY:       replay skipped entirely (agent._do_replay never called)
    B) FORWARD_REPLAY:  replay_diversity_enabled=False -> _do_replay calls
                        agent.hippocampal.replay() (plain forward)
    C) BALANCED_REPLAY: replay_diversity_enabled=True -> _do_replay calls
                        agent.hippocampal.diverse_replay(mode="auto")
                        (30% reverse, 20% random, 50% forward)

  Phase 2 (consolidation test): identical greedy (epsilon=0) test phase to
  EXQ-244a. PASS criterion unchanged: BALANCED condition maintains
  action_entropy from Phase 1 better than FORWARD condition in >= 3/5 seeds.
    entropy_retention = phase2_entropy / phase1_entropy
    PASS if mean(retention_BALANCED) > mean(retention_FORWARD) AND
         BALANCED > FORWARD in >= 3/5 seeds
    AND non-degeneracy preconditions (see NON_DEGENERACY_PRECONDITIONS below)
    are all met -- a degenerate PASS self-routes to FAIL.

Mechanism under test:
  MECH-165: offline replay must sample trajectory-diverse content to maintain
  multi-strategy viability. Reverse replay and balanced scheduling prevent
  monostrategy convergence in E1's world model.

claim_ids: [MECH-165]
experiment_purpose: diagnostic
architecture_epoch: ree_hybrid_guardrails_v1
supersedes: v3_exq_244a_mech165_replay_diversity_validation_v3
"""

import json
import os
import sys
import time
import random
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.latent.stack import LatentState
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from pathlib import Path  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"

# ---------- Constants ----------

BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4

EXPLORATION_EPISODES = 30      # Phase 1: exploration warmup
EXPLORATION_STEPS = 200        # Steps per exploration episode
EXPLORATION_EPSILON = 0.5      # Random action probability during Phase 1

CONSOLIDATION_EPISODES = 20    # Phase 2: consolidation + test
CONSOLIDATION_STEPS = 200      # Steps per consolidation episode

# Fix #4: doubled from EXQ-244a's 5 SWS cycles / ~25 diverse_replay rolls to
# 10 quiescent fires / ~50 rolls, to give replay content more opportunity to
# move Phase-2 behavior. e3_steps_per_tick defaults to 10 (ree_core/heartbeat/
# clock.py), so 100 clock ticks with no salient events yields exactly 10
# e3_quiescent fires (deterministic -- no salient event ever marked during
# the offline consolidation window).
E3_STEPS_PER_TICK = 10
NUM_QUIESCENT_FIRES_TARGET = 10
NUM_CONSOLIDATION_TICKS = E3_STEPS_PER_TICK * NUM_QUIESCENT_FIRES_TARGET

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
        # NOTE: E3_STEPS_PER_TICK below is NOT passed here -- REEConfig.from_dims
        # has no explicit e3_steps_per_tick parameter and silently swallows it
        # into **kwargs (the documented reeconfig-from-dims-silent-kwargs
        # hazard). HeartbeatConfig.e3_steps_per_tick already defaults to 10
        # (ree_core/utils/config.py:2399), which is exactly E3_STEPS_PER_TICK,
        # so the two stay in sync by relying on the (asserted below) default
        # rather than by an ineffective explicit pass-through.
    )
    assert cfg.heartbeat.e3_steps_per_tick == E3_STEPS_PER_TICK, (
        f"HeartbeatConfig default e3_steps_per_tick={cfg.heartbeat.e3_steps_per_tick} "
        f"no longer matches this driver's E3_STEPS_PER_TICK={E3_STEPS_PER_TICK} -- "
        "the quiescent-fire-count math in run_offline_consolidation()'s docstring "
        "and NUM_CONSOLIDATION_TICKS assume they match."
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
    epsilon: float,
    *,
    progress_label: str,
    progress_offset: int,
    progress_total: int,
) -> Dict:
    """Phase 1: exploration warmup. Returns action entropy.

    Fix #1: NO "final flush" agent.reset() after this function returns --
    the caller must leave agent.theta_buffer / agent._current_latent
    populated with this phase's last sensed content so consolidation can
    read real Phase-1 material instead of None.
    """
    action_counts: Counter = Counter()
    total_reward = 0.0

    for ep in range(EXPLORATION_EPISODES):
        global_ep = progress_offset + ep + 1
        if global_ep % 5 == 0 or global_ep == progress_total:
            print(f"  [train] {progress_label} phase1 ep {global_ep}/{progress_total}", flush=True)
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

    # NOTE: deliberately NO "final flush" agent.reset() here (EXQ-244a's bug).
    # theta_buffer.recent and agent._current_latent stay populated with the
    # last sensed step of this phase.

    entropy = compute_action_entropy(action_counts)
    return {
        "action_entropy": entropy,
        "total_reward": total_reward,
        "action_distribution": dict(action_counts),
        "exploration_buffer_size": len(agent.hippocampal._exploration_buffer),
    }


def run_offline_consolidation(agent: REEAgent, num_ticks: int, condition: str) -> Dict:
    """Drive offline consolidation through the PRODUCTION replay path.

    Fix #2: no direct agent.hippocampal.replay()/diverse_replay() calls, and
    no hand-rolled "if recent is not None" gate. Instead, advance the real
    multi-rate clock (agent.clock.advance()) with no new sensing -- pure
    "sleep" ticks -- and call agent._do_replay(...) on every real
    e3_quiescent tick, exactly as agent.act()/act_with_split_obs() does
    internally during a waking quiescent cycle (MECH-092). No salient event
    is ever marked during this window, so e3_quiescent fires deterministically
    every E3_STEPS_PER_TICK ticks (ree_core/heartbeat/clock.py:advance()).

    agent._do_replay's `latent_state` parameter is accepted but unused inside
    the method (it reads agent.theta_buffer.recent internally) -- passed here
    for interface fidelity with the production call sites in agent.act().

    NO_REPLAY: _do_replay is never called at all (ticks still advance, so
    NO_REPLAY and the other conditions spend an identical number of "sleep"
    ticks -- only the replay call is the manipulation).
    """
    agent.enter_sws_mode()

    reverse_before = agent.hippocampal._diverse_replay_mode_counts["reverse"]
    calls_before = agent.hippocampal._diverse_replay_mode_counts["calls"]

    quiescent_fires = 0
    for _ in range(num_ticks):
        ticks = agent.clock.advance()
        if ticks.get("e3_quiescent", False):
            quiescent_fires += 1
            if condition != "NO_REPLAY":
                agent._do_replay(agent._current_latent)

    agent.exit_sleep_mode()

    reverse_replayed = agent.hippocampal._diverse_replay_mode_counts["reverse"] - reverse_before
    diverse_replay_calls = agent.hippocampal._diverse_replay_mode_counts["calls"] - calls_before

    return {
        "num_ticks": num_ticks,
        "quiescent_fires": quiescent_fires,
        "reverse_replayed": reverse_replayed,
        "diverse_replay_calls": diverse_replay_calls,
    }


def run_phase2_test(
    agent: REEAgent,
    env: CausalGridWorldV2,
    *,
    progress_label: str,
    progress_offset: int,
    progress_total: int,
) -> Dict:
    """Phase 2: test with no exploration (epsilon=0). Returns action entropy."""
    action_counts: Counter = Counter()
    total_reward = 0.0

    for ep in range(CONSOLIDATION_EPISODES):
        global_ep = progress_offset + ep + 1
        if global_ep % 5 == 0 or global_ep == progress_total:
            print(f"  [train] {progress_label} phase2 ep {global_ep}/{progress_total}", flush=True)
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


# --- Opt-in driver-owned env seed (default OFF) ---------------------------
# Carried over verbatim from EXQ-244a: CausalGridWorldV2 forwards to
# CausalGridWorld.__init__, whose only RNG is np.random.default_rng(seed).
# np.random.default_rng(None) takes OS entropy AT CONSTRUCTION and does NOT
# consume the numpy global RNG, so this driver's np.random.seed(...) /
# torch.manual_seed(...) never reaches its env. _ENV_SEED_BASE stays None
# unless --env-seed is passed; None hands seed=None straight to the factory.
_ENV_SEED_BASE = None                 # None = OS entropy (the landed default)
_ENV_SEED_STATE = {"n": 0}            # per-CONSTRUCTION counter, not a knob


def _next_env_seed():
    if _ENV_SEED_BASE is None:
        return None
    idx = _ENV_SEED_STATE["n"]
    _ENV_SEED_STATE["n"] = idx + 1
    return int(_ENV_SEED_BASE) * 1000000 + 2 * 100000 + idx


def _env_seed_from_argv(argv):
    for i, a in enumerate(argv):
        if a == "--env-seed" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--env-seed="):
            return int(a.split("=", 1)[1])
    return None


def run_condition(condition: str, seed: int) -> Dict:
    """Run one condition for one seed. Returns the result row and the built
    agent (caller uses the agent for z_goal-stream manifest recording).

    Progress instrumentation: prints a "Seed S Condition C" boundary line
    (resets the runner's per-run episode counter), [train] ep N/M lines
    spanning BOTH phases of this seed x condition unit (denominator M =
    EXPLORATION_EPISODES + CONSOLIDATION_EPISODES, per the runner's
    multi-phase-run convention), and exactly one `verdict: PASS/FAIL` line
    at the end -- a per-cell COMPLETION marker (did this cell run and
    produce a finite, sane readout), NOT the experiment's overall
    scientific verdict, which is computed once in main() from the full
    seed x condition grid and written to the manifest's `outcome` field.
    """
    print(f"Seed {seed} Condition {condition}")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg = make_config(condition)
    agent = REEAgent(cfg)
    env = CausalGridWorldV2(seed=_next_env_seed())

    progress_total = EXPLORATION_EPISODES + CONSOLIDATION_EPISODES

    # Phase 1: exploration
    phase1 = run_phase1_exploration(
        agent, env, EXPLORATION_EPSILON,
        progress_label=f"seed={seed} {condition}",
        progress_offset=0,
        progress_total=progress_total,
    )
    phase1_entropy = phase1["action_entropy"]

    # Offline consolidation (production replay path)
    consolidation = run_offline_consolidation(agent, NUM_CONSOLIDATION_TICKS, condition)

    # Phase 2: test (no exploration)
    phase2 = run_phase2_test(
        agent, env,
        progress_label=f"seed={seed} {condition}",
        progress_offset=EXPLORATION_EPISODES,
        progress_total=progress_total,
    )
    phase2_entropy = phase2["action_entropy"]

    # Compute retention
    if phase1_entropy > 0:
        entropy_retention = phase2_entropy / phase1_entropy
    else:
        entropy_retention = 0.0

    cell_completed_cleanly = (
        np.isfinite(phase1_entropy) and np.isfinite(phase2_entropy)
        and np.isfinite(entropy_retention)
    )
    print(f"verdict: {'PASS' if cell_completed_cleanly else 'FAIL'}")

    row = {
        "condition": condition,
        "seed": seed,
        "phase1_entropy": phase1_entropy,
        "phase2_entropy": phase2_entropy,
        "entropy_retention": entropy_retention,
        "phase1_action_dist": phase1["action_distribution"],
        "phase2_action_dist": phase2["action_distribution"],
        "exploration_buffer_size": phase1["exploration_buffer_size"],
        "consolidation_metrics": consolidation,
    }
    return row, agent


def main(dry_run: bool = False):
    # Smoke/dry-run uses the same functions with module-level constants
    # temporarily overridden -- keeps the code path identical to a real run.
    global EXPLORATION_EPISODES, CONSOLIDATION_EPISODES, NUM_CONSOLIDATION_TICKS
    t0 = time.time()

    n_episodes_explore = 2 if dry_run else EXPLORATION_EPISODES
    n_episodes_consolidation = 2 if dry_run else CONSOLIDATION_EPISODES
    n_ticks = 20 if dry_run else NUM_CONSOLIDATION_TICKS
    n_seeds = [SEEDS[0]] if dry_run else SEEDS

    EXPLORATION_EPISODES = n_episodes_explore
    CONSOLIDATION_EPISODES = n_episodes_consolidation
    NUM_CONSOLIDATION_TICKS = n_ticks

    print(f"EXQ-244b: MECH-165 replay diversity validation (corrected)")
    print(f"Conditions: {CONDITIONS}")
    print(f"Seeds: {n_seeds}")
    print(f"Phase 1: {EXPLORATION_EPISODES} episodes x {EXPLORATION_STEPS} steps (epsilon={EXPLORATION_EPSILON})")
    print(f"Consolidation: {NUM_CONSOLIDATION_TICKS} offline clock ticks (~{NUM_CONSOLIDATION_TICKS // E3_STEPS_PER_TICK} quiescent fires)")
    print(f"Phase 2: {CONSOLIDATION_EPISODES} episodes x {CONSOLIDATION_STEPS} steps (greedy)")
    print()

    all_results: List[Dict] = []
    last_agent = None

    for condition in CONDITIONS:
        print(f"--- Condition: {condition} ---")
        for seed in n_seeds:
            # run_condition() itself prints the "Seed S Condition C" boundary
            # line, [train] ep N/M progress lines, and the per-cell
            # `verdict: PASS/FAIL` completion marker (progress instrumentation
            # -- distinct from the overall scientific outcome computed below).
            result, agent = run_condition(condition, seed)
            all_results.append(result)
            last_agent = agent
            print(f"  seed={seed} retention={result['entropy_retention']:.3f} "
                  f"(P1={result['phase1_entropy']:.3f}, P2={result['phase2_entropy']:.3f}) "
                  f"quiescent_fires={result['consolidation_metrics']['quiescent_fires']} "
                  f"reverse_replayed={result['consolidation_metrics']['reverse_replayed']}")

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

    balanced_rows = [r for r in all_results if r["condition"] == "BALANCED_REPLAY"]
    forward_rows = [r for r in all_results if r["condition"] == "FORWARD_REPLAY"]
    no_replay_rows = [r for r in all_results if r["condition"] == "NO_REPLAY"]

    # ---- NON-DEGENERACY PRECONDITIONS (mandatory per claim what_would_answer) ----
    # Precondition 1: reverse_replayed > 0 in at least one BALANCED_REPLAY seed.
    balanced_reverse_counts = [
        r["consolidation_metrics"]["reverse_replayed"] for r in balanced_rows
    ]
    precondition_1_reverse_fired = any(c > 0 for c in balanced_reverse_counts)

    # Precondition 2: FORWARD_REPLAY must differ from NO_REPLAY in at least one seed.
    forward_retentions = [r["entropy_retention"] for r in forward_rows]
    no_replay_retentions = [r["entropy_retention"] for r in no_replay_rows]
    forward_vs_no_replay_identical = all(
        abs(f - n) < 1e-12 for f, n in zip(forward_retentions, no_replay_retentions)
    )
    precondition_2_forward_differs = not forward_vs_no_replay_identical

    # Precondition 3: exploration buffer populated by genuine alt-strategy source
    # material (checked structurally -- unchanged from EXQ-244a's design, which
    # already only records epsilon-random actions via _record_exploration_action,
    # never the dominant/exploit-branch action).
    exploration_buffer_populated = all(
        r["exploration_buffer_size"] > 0 for r in balanced_rows
    )

    # Precondition 4: consolidation ran through the production replay path
    # (agent._do_replay -> hippocampal.diverse_replay/replay), verified by the
    # quiescent_fires counter being > 0 in every non-NO_REPLAY row (structural
    # proof the loop actually reached the production call site).
    production_path_exercised = all(
        r["consolidation_metrics"]["quiescent_fires"] > 0
        for r in (balanced_rows + forward_rows)
    )

    preconditions = [
        {
            "name": "reverse_replay_fired_at_least_one_balanced_seed",
            "description": "diverse_replay's reverse mode actually fired (agent.hippocampal._diverse_replay_mode_counts['reverse'] delta > 0) in >=1 BALANCED_REPLAY seed",
            "measured": max(balanced_reverse_counts) if balanced_reverse_counts else 0,
            "threshold": 1,
            "direction": "lower",
            "met": precondition_1_reverse_fired,
        },
        {
            "name": "forward_replay_differs_from_no_replay",
            "description": "FORWARD_REPLAY entropy_retention is not byte-identical to NO_REPLAY across all seeds",
            "measured": float(max(abs(f - n) for f, n in zip(forward_retentions, no_replay_retentions))) if forward_retentions else 0.0,
            "threshold": 1e-12,
            "direction": "lower",
            "met": precondition_2_forward_differs,
        },
        {
            "name": "exploration_buffer_populated_by_alt_strategy_source",
            "description": "exploration_buffer_size > 0 in every BALANCED_REPLAY seed (epsilon-random actions only, not the dominant policy trajectory)",
            "measured": min(r["exploration_buffer_size"] for r in balanced_rows) if balanced_rows else 0,
            "threshold": 1,
            "direction": "lower",
            "met": exploration_buffer_populated,
        },
        {
            "name": "production_replay_path_exercised",
            "description": "agent._do_replay was reached on a real e3_quiescent tick (quiescent_fires > 0) in every FORWARD_REPLAY/BALANCED_REPLAY row",
            "measured": min(
                (r["consolidation_metrics"]["quiescent_fires"] for r in (balanced_rows + forward_rows)),
                default=0,
            ),
            "threshold": 1,
            "direction": "lower",
            "met": production_path_exercised,
        },
    ]
    all_preconditions_met = all(p["met"] for p in preconditions)

    # ---- PASS criterion (unchanged from EXQ-244a) ----
    balanced_rets = condition_stats["BALANCED_REPLAY"]["retentions"]
    forward_rets = condition_stats["FORWARD_REPLAY"]["retentions"]

    seeds_balanced_wins = sum(
        1 for b, f in zip(balanced_rets, forward_rets) if b > f
    )
    pass_criterion = seeds_balanced_wins >= (1 if dry_run else PASS_SEED_MAJORITY)
    mean_balanced = condition_stats["BALANCED_REPLAY"]["mean_retention"]
    mean_forward = condition_stats["FORWARD_REPLAY"]["mean_retention"]
    mean_advantage = mean_balanced > mean_forward

    criteria_non_degenerate = {
        "C1_reverse_replay_fired": precondition_1_reverse_fired,
        "C2_forward_differs_from_no_replay": precondition_2_forward_differs,
        "C3_balanced_gt_forward_seed_majority": (
            0 < seeds_balanced_wins < len(balanced_rets)  # non-degenerate iff not a trivial 0/all sweep
        ) if len(balanced_rets) > 1 else True,
    }

    # A run whose non-degeneracy preconditions are unmet cannot support a PASS
    # verdict on the claim -- self-route to FAIL regardless of the raw
    # pass_criterion/mean_advantage arithmetic (fix #3: fail loudly, not a
    # silent marginal PASS on a harness that never actually exercised replay).
    if not all_preconditions_met:
        outcome = "FAIL"
        interpretation_label = "precondition_unmet"
    else:
        outcome = "PASS" if (pass_criterion and mean_advantage) else "FAIL"
        interpretation_label = "balanced_replay_improves_retention" if outcome == "PASS" else "balanced_replay_does_not_improve_retention"

    elapsed = time.time() - t0

    # Non-degeneracy self-report (Experimental Recording Standard -- applies
    # to diagnostic runs too, not only "evidence"). Load-bearing metrics:
    # reverse-firing count (must vary, not be pinned at 0) and the
    # BALANCED-vs-FORWARD retention separation per seed (must have spread).
    degeneracy = check_degeneracy({
        "balanced_reverse_replayed": balanced_reverse_counts,
        "balanced_minus_forward_retention": [
            b - f for b, f in zip(balanced_rets, forward_rets)
        ],
    })

    print()
    print(f"=== RESULTS ===")
    for cond in CONDITIONS:
        stats = condition_stats[cond]
        print(f"  {cond}: mean_retention={stats['mean_retention']:.4f} "
              f"+/- {stats['std_retention']:.4f}")
    print(f"  BALANCED > FORWARD: {seeds_balanced_wins}/{len(balanced_rets)} seeds "
          f"(need >= {PASS_SEED_MAJORITY})")
    print(f"  mean advantage: {mean_balanced:.4f} > {mean_forward:.4f} = {mean_advantage}")
    print(f"  Preconditions met: {all_preconditions_met} ({sum(p['met'] for p in preconditions)}/{len(preconditions)})")
    print(f"  Outcome: {outcome}")
    print(f"  Elapsed: {elapsed:.1f}s")
    # NOTE: deliberately NOT "verdict: ..." -- run_condition() already prints
    # exactly one `verdict: PASS/FAIL` line per seed x condition cell (the
    # runner's progress-counted convention: seeds x conditions lines total).
    # This is the aggregate SCIENTIFIC outcome, printed under a different
    # label so it is not double-counted by that convention.
    print(f"overall_outcome: {outcome}")

    # Build output
    run_id = f"v3_exq_244b_mech165_replay_diversity_validation_v3"
    output = {
        "run_id": run_id,
        "experiment_type": "v3_exq_244b_mech165_replay_diversity_validation",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": ["MECH-165"],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "conditions": CONDITIONS,
        "seeds": n_seeds,
        "condition_stats": condition_stats,
        "pass_criteria": {
            "seeds_balanced_wins": seeds_balanced_wins,
            "seeds_needed": PASS_SEED_MAJORITY,
            "mean_advantage": mean_advantage,
            "pass_criterion_met": pass_criterion,
        },
        # Combination rule: PASS requires BOTH load-bearing criteria (plain
        # AND) -- BALANCED > FORWARD in a seed majority AND BALANCED's mean
        # retention exceeds FORWARD's. Non-degeneracy preconditions above
        # additionally gate the verdict (unmet -> FAIL regardless).
        "criteria": [
            {
                "name": "C1_balanced_gt_forward_seed_majority",
                "load_bearing": True,
                "passed": bool(pass_criterion),
            },
            {
                "name": "C2_balanced_mean_gt_forward_mean",
                "load_bearing": True,
                "passed": bool(mean_advantage),
            },
        ],
        "interpretation": {
            "label": interpretation_label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
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
        "consolidation_config": {
            "num_ticks": NUM_CONSOLIDATION_TICKS,
            "e3_steps_per_tick": E3_STEPS_PER_TICK,
            "target_quiescent_fires": NUM_CONSOLIDATION_TICKS // E3_STEPS_PER_TICK,
        },
        "per_seed_results": all_results,
        "elapsed_seconds": elapsed,
        "supersedes": "v3_exq_244a_mech165_replay_diversity_validation_v3",
    }
    output.update(degeneracy)
    output["env_seed_base"] = _ENV_SEED_BASE

    # Write flat JSON to evidence directory
    evidence_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
    )
    os.makedirs(evidence_dir, exist_ok=True)
    outpath = write_flat_manifest(
        output,
        evidence_dir,
        dry_run=dry_run,
        config=output.get("config"),
        seeds=n_seeds,
        script_path=Path(__file__),
        agent=last_agent,
        json_default=str,
    )
    print(f"\nResults written to: {outpath}")
    return outcome, outpath


if __name__ == "__main__":
    _ENV_SEED_BASE = _env_seed_from_argv(sys.argv)
    _dry_run = "--dry-run" in sys.argv
    _outcome, _out_path = main(dry_run=_dry_run)
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id="v3_exq_244b_mech165_replay_diversity_validation_v3",
        dry_run=_dry_run,
    )
