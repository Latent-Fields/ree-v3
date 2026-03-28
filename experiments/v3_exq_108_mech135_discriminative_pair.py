"""
V3-EXQ-108 -- MECH-135 Governance-Grade Discriminative Pair

Claims: MECH-135

MECH-135 predicts: "During trajectory evaluation, E2 (cerebellar, z_self) must run in
parallel with E1 (cortical, z_world) so that z_world co-evolves during the planning
rollout; a frozen z_world causes E3 to evaluate goal achievement against a stale world
state."

Design (improves on EXQ-104b diagnostic):
  The key weakness of EXQ-104b for governance: C1 (variance ratio) is trivially true
  because FROZEN z_world has zero variance by construction (all rollouts score identically).
  EXQ-108 uses a cleaner discriminative test: plan selection quality.

  Given 40 candidate action sequences from the same starting state:
  - FROZEN condition: z_world frozen at t=0. All 40 sequences score identically (the
    initial proximity to goal). Plan selection is therefore random (no information).
  - E1_COE condition: z_world updated step-by-step via dynamic E1 prediction, reset per
    sequence. Scores vary across sequences. The best-scoring sequence is selected.

  Both best-selected sequences are then executed in the REAL ENVIRONMENT from the same
  starting state. Final goal proximity and resource contact are recorded.

  MECH-135 predicts: E1_COE best-selection will achieve higher real goal proximity than
  FROZEN best-selection (which is just random selection), because E1 co-evolution lets the
  planner identify action sequences that approach the goal.

Pass criteria (pre-registered):
  C1: mean(real_prox_e1coe - real_prox_frozen) >= 0.05  [across 2 seeds]
  C2: e1coe contact >= frozen contact in >= 1 of 2 seeds  [directional]
  C3: e1coe score variance >= 0.002 in each seed  [selection is non-trivial]

Outcome scoring:
  PASS -> evidence_direction: "supports"  (retain_ree)
  FAIL -> evidence_direction: "weakens"   (hybridize / retire_ree_claim consideration)
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
from ree_core.goal import GoalConfig, GoalState
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_108_mech135_discriminative_pair"
CLAIM_IDS = ["MECH-135"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
C1_PROX_DELTA_THRESHOLD = 0.05   # mean(real_prox_e1coe - real_prox_frozen) >= this
C2_SEEDS_NEEDED = 1              # e1coe contact >= frozen contact in >= N seeds
C3_VAR_THRESHOLD = 0.002         # e1coe score variance >= this per seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _build_agent(seed: int, world_dim: int, self_dim: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    env = CausalGridWorldV2(
        seed=seed, size=10, num_hazards=2, num_resources=4,
        hazard_harm=0.02, env_drift_interval=8, env_drift_prob=0.05,
        proximity_harm_scale=0.03, proximity_benefit_scale=0.04,
        proximity_approach_threshold=0.15, hazard_field_decay=0.5,
        resource_respawn_on_consume=True,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
    )
    config.latent.unified_latent_mode = False
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0: train agent
# ---------------------------------------------------------------------------

def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
) -> None:
    """Train agent with random policy (E1 + E2 only)."""
    torch.manual_seed(seed + 2000)
    random.seed(seed + 2000)
    agent.train()

    opt_e1 = optim.Adam(agent.e1.parameters(), lr=1e-3)
    opt_e2 = optim.Adam(agent.e2.parameters(), lr=1e-3)

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_loss_e1 = 0.0
        ep_loss_e2 = 0.0
        n_steps = 0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent_prev = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            _, _, done, _, obs_dict = env.step(action)

            obs_body_next = obs_dict["body_state"]
            obs_world_next = obs_dict["world_state"]
            with torch.no_grad():
                latent_next = agent.sense(obs_body_next, obs_world_next)

            # E1 update (world prediction error)
            opt_e1.zero_grad()
            total_prev = torch.cat([latent_prev.z_self, latent_prev.z_world], dim=-1)
            total_next = torch.cat([latent_next.z_self, latent_next.z_world], dim=-1)
            e1_pred, _ = agent.e1(total_prev, horizon=1)
            e1_loss = F.mse_loss(e1_pred[:, 0, :], total_next.detach())
            e1_loss.backward()
            opt_e1.step()
            ep_loss_e1 += e1_loss.item()

            # E2 update (motor-sensory prediction error on z_self)
            opt_e2.zero_grad()
            z_self_pred = agent.e2.predict_next_self(latent_prev.z_self.detach(), action)
            e2_loss = F.mse_loss(z_self_pred, latent_next.z_self.detach())
            e2_loss.backward()
            opt_e2.step()
            ep_loss_e2 += e2_loss.item()

            n_steps += 1
            if done:
                break

        if (ep + 1) % 20 == 0:
            print(
                f"  [Train] ep {ep+1}/{n_episodes} "
                f"e1_loss={ep_loss_e1/max(n_steps,1):.5f} "
                f"e2_loss={ep_loss_e2/max(n_steps,1):.5f}",
                flush=True,
            )

    agent.eval()
    print(f"  [Train] Done. {n_episodes} episodes.", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: collect goal template
# ---------------------------------------------------------------------------

def _collect_goal_template(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    max_steps: int,
) -> Tuple[torch.Tensor, str]:
    """Return (z_goal_tensor, source) where source is 'resource_contact' or 'fallback_unit_vector'."""
    torch.manual_seed(seed)
    random.seed(seed)
    _, obs_dict = env.reset()
    agent.reset()

    for _ in range(max_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        _, _, done, info, obs_dict = env.step(action)
        if info.get("transition_type", "none") == "resource":
            print(
                f"  [Phase1] Resource contact, z_world_norm={latent.z_world.norm().item():.3f}",
                flush=True,
            )
            return latent.z_world.detach(), "resource_contact"
        if done:
            _, obs_dict = env.reset()
            agent.reset()

    print("  [Phase1] WARNING: no resource contact -- using fallback unit vector", flush=True)
    z_goal = torch.randn(1, agent.config.latent.world_dim)
    z_goal = F.normalize(z_goal, dim=-1)
    return z_goal, "fallback_unit_vector"


# ---------------------------------------------------------------------------
# Phase 2: warmup state
# ---------------------------------------------------------------------------

def _get_warmup_state(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_warmup_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Walk agent for n_warmup_steps and return (z_self, z_world, warmup_actions).
    warmup_actions is stored so that _execute_in_real_env can reproduce the same
    starting state deterministically.
    """
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    latent = None
    warmup_actions = []

    for _ in range(n_warmup_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        warmup_actions.append(action_idx)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        _, _, done, _, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            latent = None
            warmup_actions = []

    if latent is None:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)

    return latent.z_self.detach(), latent.z_world.detach(), warmup_actions


# ---------------------------------------------------------------------------
# Phase 3: generate candidate sequences
# ---------------------------------------------------------------------------

def _generate_candidate_sequences(
    n_sequences: int,
    horizon: int,
    n_actions: int,
    seed: int,
) -> List[List[int]]:
    """Generate n_sequences random action index lists of length horizon."""
    torch.manual_seed(seed + 500)
    random.seed(seed + 500)
    seqs = []
    for _ in range(n_sequences):
        seq = [random.randint(0, n_actions - 1) for _ in range(horizon)]
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Phase 4: score sequences
# ---------------------------------------------------------------------------

def _score_sequence_frozen(
    z_world_start: torch.Tensor,
    goal_state: GoalState,
) -> float:
    """
    Score under FROZEN condition: z_world never changes, so all sequences score
    identically = initial proximity. Plan selection is therefore random (no info).
    """
    return float(goal_state.goal_proximity(z_world_start).item())


def _score_sequence_e1coe(
    agent: REEAgent,
    z_self_start: torch.Tensor,
    z_world_start: torch.Tensor,
    action_sequence: List[int],
    goal_state: GoalState,
    self_dim: int,
) -> float:
    """
    Score under E1_COE condition: z_world updated step-by-step via dynamic E1 prediction.

    E1 hidden state is reset before each sequence so scores are independent (each
    sequence represents a fresh imagined rollout from the same starting state).
    """
    device = agent.device
    n_actions = agent.config.e2.action_dim

    # Reset E1 hidden state: fresh rollout per sequence
    agent.e1.reset_hidden_state()

    z_self_curr = z_self_start.clone()
    z_world_curr = z_world_start.clone()

    for a_idx in action_sequence:
        action = _action_to_onehot(a_idx, n_actions, device)
        total_curr = torch.cat([z_self_curr, z_world_curr], dim=-1)
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_curr, horizon=1)
        # Extract world component: preds shape [1, 1, total_dim]
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        with torch.no_grad():
            z_self_next = agent.e2.predict_next_self(z_self_curr, action)
        z_self_curr = z_self_next
        z_world_curr = z_world_next

    return float(goal_state.goal_proximity(z_world_curr).item())


# ---------------------------------------------------------------------------
# Phase 5: execute best sequence in real env
# ---------------------------------------------------------------------------

def _execute_in_real_env(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    warmup_actions: List[int],
    action_sequence: List[int],
    goal_state: GoalState,
) -> Tuple[float, bool]:
    """
    Reproduce the warmup state using the stored warmup_actions, then execute
    action_sequence in the real environment. Returns (final_prox, resource_contacted).

    Using stored warmup_actions (rather than re-seeding and re-running random)
    guarantees bit-identical reproduction of the starting state regardless of any
    global random state changes during scoring.
    """
    device = agent.device

    # Reproduce warmup
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    latent = None

    for a_idx in warmup_actions:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action = _action_to_onehot(a_idx, env.action_dim, device)
        _, _, done, _, obs_dict = env.step(action)
        if done:
            break

    # Execute candidate sequence in real env
    resource_contacted = False
    for a_idx in action_sequence:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action = _action_to_onehot(a_idx, env.action_dim, device)
        _, _, done, info, obs_dict = env.step(action)
        if info.get("transition_type", "none") == "resource":
            resource_contacted = True
        if done:
            break

    # Final obs
    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    with torch.no_grad():
        latent_final = agent.sense(obs_body, obs_world)

    final_prox = float(goal_state.goal_proximity(latent_final.z_world).item())
    return final_prox, resource_contacted


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def run_seed(
    seed: int,
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    n_warmup_steps: int,
    goal_max_steps: int,
    c1_threshold: float,
    c3_var_threshold: float,
) -> Dict:
    print(f"\n[EXQ-108] seed={seed}", flush=True)

    agent, env = _build_agent(seed, world_dim, self_dim)

    # Phase 0: train
    print(f"[EXQ-108] Phase 0: training ({n_train_episodes} eps)...", flush=True)
    _train_agent(agent, env, seed, n_train_episodes, steps_per_episode)

    # Phase 1: goal template
    print("[EXQ-108] Phase 1: goal template...", flush=True)
    z_goal_tensor, goal_template_source = _collect_goal_template(agent, env, seed, goal_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state = GoalState(goal_config, agent.device)
    goal_state._z_goal = z_goal_tensor.to(agent.device)
    print(f"  z_goal_norm={goal_state.goal_norm():.4f} source={goal_template_source}", flush=True)

    # Phase 2: warmup state
    print("[EXQ-108] Phase 2: warmup state...", flush=True)
    z_self_0, z_world_0, warmup_actions = _get_warmup_state(agent, env, seed, n_warmup_steps)
    base_prox = float(goal_state.goal_proximity(z_world_0).item())
    print(f"  base_prox={base_prox:.4f}", flush=True)

    # Phase 3: candidate sequences
    print(f"[EXQ-108] Phase 3: generating {n_sequences} candidate sequences...", flush=True)
    seqs = _generate_candidate_sequences(n_sequences, rollout_horizon, env.action_dim, seed)

    # Phase 4: score under FROZEN and E1_COE
    print("[EXQ-108] Phase 4: scoring sequences...", flush=True)
    frozen_scores = []
    e1coe_scores = []

    for i, seq in enumerate(seqs):
        # FROZEN: all sequences score identically = base_prox
        frozen_scores.append(_score_sequence_frozen(z_world_0, goal_state))

        # E1_COE: dynamic E1 co-evolution, hidden state reset per sequence
        e1coe_score = _score_sequence_e1coe(agent, z_self_0, z_world_0, seq, goal_state, self_dim)
        e1coe_scores.append(e1coe_score)

        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{n_sequences}", flush=True)

    e1coe_scores_t = torch.tensor(e1coe_scores)
    e1coe_score_var = float(e1coe_scores_t.var().item()) if len(e1coe_scores) > 1 else 0.0
    e1coe_score_min = float(e1coe_scores_t.min().item())
    e1coe_score_max = float(e1coe_scores_t.max().item())
    e1coe_score_mean = float(e1coe_scores_t.mean().item())

    print(
        f"  E1_COE scores: min={e1coe_score_min:.4f} max={e1coe_score_max:.4f} "
        f"mean={e1coe_score_mean:.4f} var={e1coe_score_var:.6f}",
        flush=True,
    )
    print(f"  FROZEN scores: all={frozen_scores[0]:.4f} (constant)", flush=True)

    # Phase 5: select best sequences
    # FROZEN: argmax of constant array = index 0 = effectively random
    frozen_best_idx = int(torch.tensor(frozen_scores).argmax().item())
    e1coe_best_idx = int(e1coe_scores_t.argmax().item())
    print(
        f"  FROZEN best_idx={frozen_best_idx} (random) "
        f"E1_COE best_idx={e1coe_best_idx} score={e1coe_scores[e1coe_best_idx]:.4f}",
        flush=True,
    )

    # Phase 6: execute best sequences in real env
    print("[EXQ-108] Phase 6: real-env execution...", flush=True)
    agent.eval()
    real_prox_frozen, resource_frozen = _execute_in_real_env(
        agent, env, seed, warmup_actions, seqs[frozen_best_idx], goal_state
    )
    real_prox_e1coe, resource_e1coe = _execute_in_real_env(
        agent, env, seed, warmup_actions, seqs[e1coe_best_idx], goal_state
    )

    prox_delta = real_prox_e1coe - real_prox_frozen
    print(
        f"  FROZEN: real_prox={real_prox_frozen:.4f} contact={resource_frozen}",
        flush=True,
    )
    print(
        f"  E1_COE: real_prox={real_prox_e1coe:.4f} contact={resource_e1coe} "
        f"delta={prox_delta:+.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "goal_template_source": goal_template_source,
        "z_goal_norm": goal_state.goal_norm(),
        "base_prox": base_prox,
        "e1coe_score_min": e1coe_score_min,
        "e1coe_score_max": e1coe_score_max,
        "e1coe_score_mean": e1coe_score_mean,
        "e1coe_score_var": e1coe_score_var,
        "e1coe_best_score": float(e1coe_scores[e1coe_best_idx]),
        "frozen_base_score": frozen_scores[0],
        "real_prox_frozen": real_prox_frozen,
        "real_prox_e1coe": real_prox_e1coe,
        "prox_delta": prox_delta,
        "resource_frozen": resource_frozen,
        "resource_e1coe": resource_e1coe,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    seeds: List[int],
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    n_warmup_steps: int,
    goal_max_steps: int,
    c1_threshold: float,
    c3_var_threshold: float,
) -> Dict:
    seed_results = []
    for seed in seeds:
        r = run_seed(
            seed=seed,
            world_dim=world_dim,
            self_dim=self_dim,
            n_train_episodes=n_train_episodes,
            steps_per_episode=steps_per_episode,
            n_sequences=n_sequences,
            rollout_horizon=rollout_horizon,
            n_warmup_steps=n_warmup_steps,
            goal_max_steps=goal_max_steps,
            c1_threshold=c1_threshold,
            c3_var_threshold=c3_var_threshold,
        )
        seed_results.append(r)

    # Aggregate
    prox_deltas = [r["prox_delta"] for r in seed_results]
    c1_val = sum(prox_deltas) / len(prox_deltas)
    c1_pass = c1_val >= c1_threshold

    c2_per_seed = [int(r["resource_e1coe"]) >= int(r["resource_frozen"]) for r in seed_results]
    c2_seeds_passing = sum(c2_per_seed)
    c2_pass = c2_seeds_passing >= C2_SEEDS_NEEDED

    c3_per_seed = [r["e1coe_score_var"] >= c3_var_threshold for r in seed_results]
    c3_pass = all(c3_per_seed)

    print(f"\n[EXQ-108] Criteria:", flush=True)
    print(f"  C1 (mean_delta>={c1_threshold:.3f}): {c1_pass} val={c1_val:+.4f}", flush=True)
    print(f"  C2 (contact directional, >= {C2_SEEDS_NEEDED} seeds): {c2_pass} ({c2_seeds_passing}/{len(seeds)} seeds)", flush=True)
    print(f"  C3 (e1coe_var>={c3_var_threshold:.4f}): {c3_pass} {[r['e1coe_score_var'] for r in seed_results]}", flush=True)

    criteria_met = sum([c1_pass, c2_pass, c3_pass])
    status = "PASS" if (c1_pass and c2_pass and c3_pass) else "FAIL"
    evidence_direction = "supports" if status == "PASS" else "weakens"

    print(f"[EXQ-108] Status: {status}", flush=True)

    result = {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "evidence_class": "discriminative_pair",
        "evidence_direction": evidence_direction,
        "seeds": seeds,
        "world_dim": world_dim,
        "self_dim": self_dim,
        "n_train_episodes": n_train_episodes,
        "steps_per_episode": steps_per_episode,
        "n_sequences": n_sequences,
        "rollout_horizon": rollout_horizon,
        "n_warmup_steps": n_warmup_steps,
        # Pre-registered thresholds (recorded for governance)
        "registered_c1_threshold": c1_threshold,
        "registered_c3_var_threshold": c3_var_threshold,
        "registered_c2_seeds_needed": C2_SEEDS_NEEDED,
        # Per-seed metrics
        "c1_mean_prox_delta": c1_val,
        "c1_pass": bool(c1_pass),
        "c2_pass": bool(c2_pass),
        "c2_seeds_passing": c2_seeds_passing,
        "c3_pass": bool(c3_pass),
        "c3_per_seed_vars": [r["e1coe_score_var"] for r in seed_results],
        "criteria_met": criteria_met,
        "criteria_total": 3,
        "status": status,
        "verdict": status,
    }

    # Flatten per-seed metrics
    for r in seed_results:
        s = r["seed"]
        for k, v in r.items():
            if k != "seed":
                result[f"seed_{s}_{k}"] = v

    return result


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-108: MECH-135 discriminative pair (E1 co-evolution vs frozen z_world)"
    )
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--world-dim", type=int, default=32)
    parser.add_argument("--self-dim", type=int, default=32)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--n-sequences", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--goal-max-steps", type=int, default=2000)
    parser.add_argument("--c1-threshold", type=float, default=C1_PROX_DELTA_THRESHOLD)
    parser.add_argument("--c3-var-threshold", type=float, default=C3_VAR_THRESHOLD)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.smoke_test:
        n_train = 2
        steps_ep = 50
        n_sequences = 5
        horizon = 10
        warmup = 5
        goal_max = 300
        c1_thresh = -99.0
        c3_thresh = 0.0
        seeds = seeds[:1]  # single seed for speed
        print("[V3-EXQ-108] SMOKE TEST MODE", flush=True)
    else:
        n_train = args.train_episodes
        steps_ep = args.steps_per_episode
        n_sequences = args.n_sequences
        horizon = args.rollout_horizon
        warmup = args.warmup_steps
        goal_max = args.goal_max_steps
        c1_thresh = args.c1_threshold
        c3_thresh = args.c3_var_threshold

    result = run(
        seeds=seeds,
        world_dim=args.world_dim,
        self_dim=args.self_dim,
        n_train_episodes=n_train,
        steps_per_episode=steps_ep,
        n_sequences=n_sequences,
        rollout_horizon=horizon,
        n_warmup_steps=warmup,
        goal_max_steps=goal_max,
        c1_threshold=c1_thresh,
        c3_var_threshold=c3_thresh,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
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

    if args.smoke_test:
        print("[V3-EXQ-108] SMOKE TEST COMPLETE", flush=True)
        for k in ["c1_mean_prox_delta", "c1_pass", "c2_pass", "c3_pass",
                  "c3_per_seed_vars", "criteria_met"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
