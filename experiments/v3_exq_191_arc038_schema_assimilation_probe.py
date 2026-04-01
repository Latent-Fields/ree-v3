#!/opt/local/bin/python3
"""
V3-EXQ-191 -- ARC-038 Schema Assimilation Probe

Claims: ARC-038

Scientific question: Does the agent's hippocampal module show accelerated
learning (schema transfer) when presented with a structurally similar but
novel environment, compared to a naive agent encountering the same novel
environment without prior training?

ARC-038 asserts that hippocampal replay has a waking consolidation mode
(distinct from goal-directed forward sweeps) that integrates recent
experience into map geometry. If this mechanism functions, an agent that
has learned the spatial structure of one environment should transfer harm-
avoidance faster to a structurally similar but repositioned environment.

Design:
  Phase 1 (SCHEMA_PRIMED only): Train agent on Environment-1 (base layout,
    env_seed_A) for training_episodes episodes with random exploration.
    This builds a "schema" -- the hippocampal module and E1/E2/E3 learn
    the grid structure, hazard proximity dynamics, and harm avoidance.

  Phase 2 (both conditions): Evaluate on Environment-2 (env_seed_B --
    same grid size, same num_hazards/resources, different positions).
    - SCHEMA_PRIMED: agent carries over all weights from Phase 1.
    - NAIVE: fresh agent with no Phase 1 training.
    Both agents take actions via harm_eval-based selection (not random).
    Track per-episode harm rate across eval_episodes episodes.

  Measure: episodes-to-target -- the first episode block (of block_size)
    where mean harm_rate drops below target_harm_rate.

  Condition RANDOM: random actions on Environment-2 (baseline).

Pre-registered PASS criteria:
  C1: schema_episodes_to_target < naive_episodes_to_target
      (primed agent reaches target faster)
  C2: schema_final_harm_rate < naive_final_harm_rate
      (primed agent ends with lower harm over final block)
  C3: schema_episodes_to_target < eval_episodes
      (primed agent actually reaches target within budget)

Seeds: 2 experiment seeds x 2 environment seed pairs per experiment seed.

PASS: C1 and C2 and C3 all met (averaged across seeds).
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_191_arc038_schema_assimilation_probe"
CLAIM_IDS = ["ARC-038"]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _select_action_harm_avoid(
    agent: REEAgent,
    z_world: torch.Tensor,
    num_actions: int,
) -> int:
    """Pick action with lowest predicted harm (E3 harm_eval on E2 world_forward)."""
    with torch.no_grad():
        best_action = 0
        best_score = float("inf")
        for idx in range(num_actions):
            a_oh = _action_to_onehot(idx, num_actions, z_world.device)
            z_world_next = agent.e2.world_forward(z_world, a_oh)
            harm_score = agent.e3.harm_eval(z_world_next).mean().item()
            if harm_score < best_score:
                best_score = harm_score
                best_action = idx
    return best_action


def _make_env(env_seed: int, grid_size: int = 10) -> CausalGridWorldV2:
    """Create a CausalGridWorldV2 with proxy fields and standard parameters."""
    return CausalGridWorldV2(
        seed=env_seed,
        size=grid_size,
        num_hazards=3,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.3,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, self_dim: int, world_dim: int) -> REEAgent:
    """Create a fresh REEAgent configured for the given environment."""
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
    )
    return REEAgent(config)


# ------------------------------------------------------------------ #
# Training loop (Phase 1: build schema on Environment-1)               #
# ------------------------------------------------------------------ #

def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
) -> None:
    """Train agent with random exploration on given env. Builds schema."""
    action_dim = env.action_dim

    # Separate optimizer groups
    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=1e-3)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    # Stratified harm replay buffer
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 4000

    agent.train()

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for step_i in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            # Random action during training
            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            # --- Standard agent training (E1 + E2) ---
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # --- harm_eval training (stratified) ---
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.binary_cross_entropy(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_opt.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_opt.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(
                f"    [train] ep {ep+1}/{num_episodes}"
                f" harm_pos={len(harm_buf_pos)} harm_neg={len(harm_buf_neg)}",
                flush=True,
            )


# ------------------------------------------------------------------ #
# Evaluation loop (Phase 2: test on Environment-2)                     #
# ------------------------------------------------------------------ #

def _eval_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    condition_label: str,
    use_random_actions: bool = False,
) -> List[float]:
    """
    Evaluate agent on env for num_episodes, continuing to train E1/E2/harm_eval
    online (simulating ongoing learning in the new environment).

    Returns per-episode harm rates.
    """
    action_dim = env.action_dim

    # Online learning during eval (schema transfer = continued learning, not frozen)
    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=1e-3)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 4000

    agent.train()
    per_episode_harm: List[float] = []

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm_sum = 0.0
        ep_steps = 0

        for step_i in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            if use_random_actions:
                action_idx = random.randint(0, action_dim - 1)
            else:
                action_idx = _select_action_harm_avoid(
                    agent, z_world_curr, action_dim,
                )

            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            # Online training (both schema-primed and naive learn during eval)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # harm_eval online training
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.binary_cross_entropy(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_opt.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_opt.step()

            if float(harm_signal) < 0:
                ep_harm_sum += abs(float(harm_signal))
            ep_steps += 1

            if done:
                break

        harm_rate = ep_harm_sum / max(1, ep_steps)
        per_episode_harm.append(harm_rate)

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            block_start = max(0, ep - 9)
            recent_mean = sum(per_episode_harm[block_start:]) / max(1, ep + 1 - block_start)
            print(
                f"    [eval {condition_label}] ep {ep+1}/{num_episodes}"
                f" harm_rate={harm_rate:.5f} recent_10_mean={recent_mean:.5f}",
                flush=True,
            )

    return per_episode_harm


# ------------------------------------------------------------------ #
# Analysis helpers                                                     #
# ------------------------------------------------------------------ #

def _episodes_to_target(
    per_ep_harm: List[float],
    target: float,
    block_size: int,
) -> int:
    """
    Return the first episode index where the block mean drops below target.
    If never reached, return len(per_ep_harm) (= budget exhausted).
    """
    n = len(per_ep_harm)
    for i in range(block_size - 1, n):
        block = per_ep_harm[i - block_size + 1 : i + 1]
        block_mean = sum(block) / len(block)
        if block_mean < target:
            return i + 1  # episodes consumed
    return n


def _final_block_mean(per_ep_harm: List[float], block_size: int) -> float:
    """Mean harm rate of the final block_size episodes."""
    if len(per_ep_harm) < block_size:
        return sum(per_ep_harm) / max(1, len(per_ep_harm))
    block = per_ep_harm[-block_size:]
    return sum(block) / len(block)


# ------------------------------------------------------------------ #
# Single seed run                                                      #
# ------------------------------------------------------------------ #

def _run_single(
    experiment_seed: int,
    env_seed_a: int,
    env_seed_b: int,
    training_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    target_harm_rate: float,
    block_size: int,
    self_dim: int,
    world_dim: int,
) -> Dict:
    torch.manual_seed(experiment_seed)
    random.seed(experiment_seed)

    grid_size = 10

    # ---- SCHEMA_PRIMED condition ----
    print(f"  [SCHEMA_PRIMED] Training on env_seed_a={env_seed_a}...", flush=True)
    env_a = _make_env(env_seed_a, grid_size)
    primed_agent = _make_agent(env_a, self_dim, world_dim)
    _train_agent(primed_agent, env_a, training_episodes, steps_per_episode)

    print(f"  [SCHEMA_PRIMED] Evaluating on env_seed_b={env_seed_b}...", flush=True)
    env_b_primed = _make_env(env_seed_b, grid_size)
    primed_harm = _eval_agent(
        primed_agent, env_b_primed, eval_episodes, steps_per_episode,
        condition_label="SCHEMA_PRIMED",
    )

    # ---- NAIVE condition ----
    # Reset RNG so naive agent gets same initial weights as primed would have
    # if it had not trained -- use a derived seed to ensure independence.
    torch.manual_seed(experiment_seed + 10000)
    random.seed(experiment_seed + 10000)

    print(f"  [NAIVE] Evaluating on env_seed_b={env_seed_b} (no pre-training)...",
          flush=True)
    env_b_naive = _make_env(env_seed_b, grid_size)
    naive_agent = _make_agent(env_b_naive, self_dim, world_dim)
    naive_harm = _eval_agent(
        naive_agent, env_b_naive, eval_episodes, steps_per_episode,
        condition_label="NAIVE",
    )

    # ---- RANDOM baseline ----
    torch.manual_seed(experiment_seed + 20000)
    random.seed(experiment_seed + 20000)

    print(f"  [RANDOM] Evaluating on env_seed_b={env_seed_b}...", flush=True)
    env_b_random = _make_env(env_seed_b, grid_size)
    random_agent = _make_agent(env_b_random, self_dim, world_dim)
    random_harm = _eval_agent(
        random_agent, env_b_random, eval_episodes, steps_per_episode,
        condition_label="RANDOM",
        use_random_actions=True,
    )

    # ---- Compute metrics ----
    primed_ett = _episodes_to_target(primed_harm, target_harm_rate, block_size)
    naive_ett = _episodes_to_target(naive_harm, target_harm_rate, block_size)
    random_ett = _episodes_to_target(random_harm, target_harm_rate, block_size)

    primed_final = _final_block_mean(primed_harm, block_size)
    naive_final = _final_block_mean(naive_harm, block_size)
    random_final = _final_block_mean(random_harm, block_size)

    return {
        "experiment_seed": experiment_seed,
        "env_seed_a": env_seed_a,
        "env_seed_b": env_seed_b,
        "primed_episodes_to_target": primed_ett,
        "naive_episodes_to_target": naive_ett,
        "random_episodes_to_target": random_ett,
        "primed_final_harm_rate": float(primed_final),
        "naive_final_harm_rate": float(naive_final),
        "random_final_harm_rate": float(random_final),
        "primed_mean_harm_rate": float(sum(primed_harm) / max(1, len(primed_harm))),
        "naive_mean_harm_rate": float(sum(naive_harm) / max(1, len(naive_harm))),
        "random_mean_harm_rate": float(sum(random_harm) / max(1, len(random_harm))),
        "primed_harm_curve": [round(h, 6) for h in primed_harm],
        "naive_harm_curve": [round(h, 6) for h in naive_harm],
        "random_harm_curve": [round(h, 6) for h in random_harm],
    }


# ------------------------------------------------------------------ #
# Aggregation and output                                               #
# ------------------------------------------------------------------ #

def run(
    seeds: Tuple = (42, 7),
    env_seed_pairs: Tuple = ((100, 200), (300, 400)),
    training_episodes: int = 200,
    eval_episodes: int = 200,
    steps_per_episode: int = 200,
    target_harm_rate: float = 0.002,
    block_size: int = 20,
    self_dim: int = 16,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    all_results: List[Dict] = []

    for seed in seeds:
        for env_seed_a, env_seed_b in env_seed_pairs:
            print(
                f"\n[V3-EXQ-191] seed={seed} env_a={env_seed_a} env_b={env_seed_b}"
                f" train={training_episodes} eval={eval_episodes}"
                f" steps={steps_per_episode}",
                flush=True,
            )
            r = _run_single(
                experiment_seed=seed,
                env_seed_a=env_seed_a,
                env_seed_b=env_seed_b,
                training_episodes=training_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                target_harm_rate=target_harm_rate,
                block_size=block_size,
                self_dim=self_dim,
                world_dim=world_dim,
            )
            all_results.append(r)

    n = len(all_results)

    # Aggregate
    avg_primed_ett = sum(r["primed_episodes_to_target"] for r in all_results) / n
    avg_naive_ett = sum(r["naive_episodes_to_target"] for r in all_results) / n
    avg_random_ett = sum(r["random_episodes_to_target"] for r in all_results) / n

    avg_primed_final = sum(r["primed_final_harm_rate"] for r in all_results) / n
    avg_naive_final = sum(r["naive_final_harm_rate"] for r in all_results) / n
    avg_random_final = sum(r["random_final_harm_rate"] for r in all_results) / n

    avg_primed_mean = sum(r["primed_mean_harm_rate"] for r in all_results) / n
    avg_naive_mean = sum(r["naive_mean_harm_rate"] for r in all_results) / n
    avg_random_mean = sum(r["random_mean_harm_rate"] for r in all_results) / n

    # PASS criteria
    c1_pass = avg_primed_ett < avg_naive_ett
    c2_pass = avg_primed_final < avg_naive_final
    c3_pass = avg_primed_ett < eval_episodes

    all_pass = c1_pass and c2_pass and c3_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass])
    status = "PASS" if all_pass else "FAIL"

    # Speedup ratio (how much faster primed reaches target vs naive)
    speedup_ratio = (
        avg_naive_ett / max(1.0, avg_primed_ett)
        if avg_primed_ett > 0 else 0.0
    )

    if all_pass:
        decision = "retain_ree"
    elif criteria_met >= 2:
        decision = "hybridize"
    elif c3_pass:
        decision = "inconclusive"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-191] Final results:", flush=True)
    print(
        f"  SCHEMA_PRIMED: episodes_to_target={avg_primed_ett:.1f}"
        f" final_harm={avg_primed_final:.6f}"
        f" mean_harm={avg_primed_mean:.6f}",
        flush=True,
    )
    print(
        f"  NAIVE:         episodes_to_target={avg_naive_ett:.1f}"
        f" final_harm={avg_naive_final:.6f}"
        f" mean_harm={avg_naive_mean:.6f}",
        flush=True,
    )
    print(
        f"  RANDOM:        episodes_to_target={avg_random_ett:.1f}"
        f" final_harm={avg_random_final:.6f}"
        f" mean_harm={avg_random_mean:.6f}",
        flush=True,
    )
    print(
        f"  speedup_ratio={speedup_ratio:.2f}x",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/3)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: primed_ett={avg_primed_ett:.1f}"
            f" >= naive_ett={avg_naive_ett:.1f}."
            " Schema-primed agent did not reach target faster."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: primed_final={avg_primed_final:.6f}"
            f" >= naive_final={avg_naive_final:.6f}."
            " Schema-primed agent did not achieve lower final harm."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: primed_ett={avg_primed_ett:.1f}"
            f" >= eval_budget={eval_episodes}."
            " Schema-primed agent never reached target harm rate."
        )
    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_seed_rows = "\n".join(
        f"  seed={r['experiment_seed']} env_a={r['env_seed_a']} env_b={r['env_seed_b']}:"
        f" primed_ett={r['primed_episodes_to_target']}"
        f" naive_ett={r['naive_episodes_to_target']}"
        f" primed_final={r['primed_final_harm_rate']:.6f}"
        f" naive_final={r['naive_final_harm_rate']:.6f}"
        for r in all_results
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-191 -- ARC-038 Schema Assimilation Probe\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-038\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Env seed pairs:** {list(env_seed_pairs)}\n\n"
        f"## Design\n\n"
        f"Schema transfer test: agent trained on Environment-1 (schema building),\n"
        f"then evaluated on Environment-2 (novel but structurally similar). Compared\n"
        f"to naive agent seeing Environment-2 for the first time.\n"
        f"Both conditions use harm_eval-based action selection and online learning\n"
        f"during evaluation. Measure: episodes to reach target harm rate.\n\n"
        f"## Results\n\n"
        f"| Condition | Eps to Target | Final Harm Rate | Mean Harm Rate |\n"
        f"|---|---|---|---|\n"
        f"| SCHEMA_PRIMED | {avg_primed_ett:.1f} | {avg_primed_final:.6f}"
        f" | {avg_primed_mean:.6f} |\n"
        f"| NAIVE | {avg_naive_ett:.1f} | {avg_naive_final:.6f}"
        f" | {avg_naive_mean:.6f} |\n"
        f"| RANDOM | {avg_random_ett:.1f} | {avg_random_final:.6f}"
        f" | {avg_random_mean:.6f} |\n\n"
        f"**Speedup ratio (naive_ett / primed_ett):** {speedup_ratio:.2f}x\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Detail |\n"
        f"|---|---|---|\n"
        f"| C1: primed_ett < naive_ett | {'PASS' if c1_pass else 'FAIL'}"
        f" | {avg_primed_ett:.1f} vs {avg_naive_ett:.1f} |\n"
        f"| C2: primed_final < naive_final | {'PASS' if c2_pass else 'FAIL'}"
        f" | {avg_primed_final:.6f} vs {avg_naive_final:.6f} |\n"
        f"| C3: primed_ett < budget | {'PASS' if c3_pass else 'FAIL'}"
        f" | {avg_primed_ett:.1f} vs {eval_episodes} |\n\n"
        f"Criteria met: {criteria_met}/3 -> **{status}**\n\n"
        f"## Per-Seed\n\n{per_seed_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "primed_episodes_to_target":    float(avg_primed_ett),
        "naive_episodes_to_target":     float(avg_naive_ett),
        "random_episodes_to_target":    float(avg_random_ett),
        "primed_final_harm_rate":       float(avg_primed_final),
        "naive_final_harm_rate":        float(avg_naive_final),
        "random_final_harm_rate":       float(avg_random_final),
        "primed_mean_harm_rate":        float(avg_primed_mean),
        "naive_mean_harm_rate":         float(avg_naive_mean),
        "random_mean_harm_rate":        float(avg_random_mean),
        "speedup_ratio":                float(speedup_ratio),
        "target_harm_rate":             float(target_harm_rate),
        "block_size":                   float(block_size),
        "training_episodes":            float(training_episodes),
        "eval_episodes":                float(eval_episodes),
        "n_runs":                       float(n),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    # Per-seed detail (without full harm curves for manifest size)
    per_seed_results = []
    for r in all_results:
        per_seed_results.append({
            "experiment_seed": r["experiment_seed"],
            "env_seed_a": r["env_seed_a"],
            "env_seed_b": r["env_seed_b"],
            "primed_episodes_to_target": r["primed_episodes_to_target"],
            "naive_episodes_to_target": r["naive_episodes_to_target"],
            "random_episodes_to_target": r["random_episodes_to_target"],
            "primed_final_harm_rate": r["primed_final_harm_rate"],
            "naive_final_harm_rate": r["naive_final_harm_rate"],
            "random_final_harm_rate": r["random_final_harm_rate"],
            "primed_mean_harm_rate": r["primed_mean_harm_rate"],
            "naive_mean_harm_rate": r["naive_mean_harm_rate"],
            "random_mean_harm_rate": r["random_mean_harm_rate"],
        })

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "per_seed_results": per_seed_results,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",       type=int, nargs="+", default=[42, 7])
    parser.add_argument("--train-eps",   type=int, default=200)
    parser.add_argument("--eval-eps",    type=int, default=200)
    parser.add_argument("--steps",       type=int, default=200)
    parser.add_argument("--target-harm", type=float, default=0.002)
    parser.add_argument("--block-size",  type=int, default=20)
    parser.add_argument("--dry-run",     action="store_true",
                        help="Run minimal config for smoke test")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds = [42]
        args.train_eps = 3
        args.eval_eps = 5
        env_seed_pairs = ((100, 200),)
        print("[DRY-RUN] 1 seed, 1 env pair, 3 train, 5 eval", flush=True)
    else:
        env_seed_pairs = ((100, 200), (300, 400))

    result = run(
        seeds=tuple(args.seeds),
        env_seed_pairs=env_seed_pairs,
        training_episodes=args.train_eps,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        target_harm_rate=args.target_harm,
        block_size=args.block_size,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output file.", flush=True)
        print(f"Status: {result['status']}", flush=True)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
        sys.exit(0)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
