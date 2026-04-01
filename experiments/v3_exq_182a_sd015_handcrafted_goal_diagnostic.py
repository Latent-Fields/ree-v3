#!/opt/local/bin/python3
"""
V3-EXQ-182a -- SD-015 Handcrafted Goal Cue Diagnostic

Claims: SD-015

=== SCIENTIFIC QUESTION ===

If E3 gets a *perfect, handcrafted* resource proximity signal as the goal
score (bypassing all learned representation), can it exploit it behaviorally?

This is a ceiling test: if a perfect oracle goal signal does not produce
meaningful resource collection, the bottleneck is NOT in goal representation
(z_goal learning, encoder quality, etc.) but in the action-selection or
harm-avoidance tradeoff itself.

=== DESIGN ===

Two conditions:
  GOAL_HANDCRAFTED: Action selection uses oracle resource proximity.
    For each candidate action, compute the agent's next position using env
    movement rules, compute manhattan_dist to nearest resource from that
    position, score = 1/(1+dist). Combined with harm signal:
      score(a) = -lambda_goal * oracle_prox(a) + lambda_harm * harm_at_next(a)
    Pick action with lowest combined score (most goal, least harm).

  GOAL_ABSENT: Random action selection (baseline, no goal signal).

Warmup: 200 episodes (standard REE training with random actions)
Eval:   100 episodes per condition per seed
Steps:  200 per episode
Seeds:  [42, 7, 13]

=== PASS CRITERIA ===

C1: benefit_ratio >= 1.3   (handcrafted beats absent by 30%)
C2: mean_benefit_handcrafted > 0.5   (meaningful resource collection)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_182a_sd015_handcrafted_goal_diagnostic"
CLAIM_IDS = ["SD-015"]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(0)
    return t


# Movement deltas matching CausalGridWorld.ACTIONS
# 0: up (-1,0), 1: down (+1,0), 2: left (0,-1), 3: right (0,+1), 4: stay (0,0)
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0),
}


def _next_position(env, action_idx: int) -> Tuple[int, int]:
    """Compute agent's next position given action, respecting walls."""
    dx, dy = ACTION_DELTAS[action_idx]
    nx = env.agent_x + dx
    ny = env.agent_y + dy
    # If wall, agent stays in place
    if env.grid[nx, ny] == env.ENTITY_TYPES["wall"]:
        return env.agent_x, env.agent_y
    return nx, ny


def _manhattan_dist_to_nearest_resource(x: int, y: int, resources) -> float:
    """Manhattan distance from (x, y) to nearest resource. Returns inf if none."""
    if not resources:
        return float("inf")
    return min(abs(x - int(r[0])) + abs(y - int(r[1])) for r in resources)


def _oracle_proximity(x: int, y: int, resources) -> float:
    """Perfect oracle resource proximity: 1/(1+manhattan_dist)."""
    dist = _manhattan_dist_to_nearest_resource(x, y, resources)
    if dist == float("inf"):
        return 0.0
    return 1.0 / (1.0 + dist)


def _harm_at_position(x: int, y: int, env) -> float:
    """Estimate harm at position from hazard_field (proxy gradient).
    Returns value in [0, 1] range -- higher means more harmful."""
    if env.use_proxy_fields:
        return float(env.hazard_field[x, y])
    # Fallback: binary check for hazard entity
    if env.grid[x, y] == env.ENTITY_TYPES["hazard"]:
        return 1.0
    return 0.0


def _oracle_goal_action(env, lambda_goal: float = 2.0,
                         lambda_harm: float = 1.0) -> int:
    """Select action using oracle goal proximity + harm avoidance.

    score(a) = -lambda_goal * oracle_prox(next_pos(a))
               + lambda_harm * hazard_field(next_pos(a))

    Lower score = better (more goal, less harm). Returns best action index.
    """
    best_action = 0
    best_score = float("inf")
    num_actions = len(ACTION_DELTAS)

    for a_idx in range(num_actions):
        nx, ny = _next_position(env, a_idx)
        goal_prox = _oracle_proximity(nx, ny, env.resources)
        harm_val = _harm_at_position(nx, ny, env)
        score = -lambda_goal * goal_prox + lambda_harm * harm_val
        if score < best_score:
            best_score = score
            best_action = a_idx

    return best_action


def _resource_proximity(env) -> float:
    """Current oracle resource proximity at agent position."""
    return _oracle_proximity(env.agent_x, env.agent_y, env.resources)


# ------------------------------------------------------------------ #
# Main run function                                                     #
# ------------------------------------------------------------------ #

def _run_single(
    seed: int,
    goal_handcrafted: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    lambda_goal: float,
    lambda_harm: float,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "GOAL_HANDCRAFTED" if goal_handcrafted else "GOAL_ABSENT"

    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )

    action_dim = env.action_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
    )

    agent = REEAgent(config)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 4000

    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer     = optim.Adam(standard_params, lr=lr)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "resource": 0,
        "benefit_approach": 0,
    }

    agent.train()

    # ---- WARMUP: random actions, train agent normally ----
    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            # Harm buffer for stratified harm_eval training
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF // 2:
                    harm_buf_pos = harm_buf_pos[-(MAX_BUF // 2):]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF // 2:
                    harm_buf_neg = harm_buf_neg[-(MAX_BUF // 2):]

            # Standard agent training (E1 + E2)
            e1_loss    = agent.compute_prediction_loss()
            e2_loss    = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # Stratified harm_eval training
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos  = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg  = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b    = torch.cat([zw_pos, zw_neg], dim=0)
                target  = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
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

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [warmup] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" resource={counts['resource']}"
                f" harm_pos={len(harm_buf_pos)} harm_neg={len(harm_buf_neg)}",
                flush=True,
            )

    # ---- EVAL ----
    agent.eval()

    benefit_per_ep: List[float] = []
    oracle_prox_vals: List[float] = []
    harm_per_ep: List[float] = []

    for eval_ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_benefit = 0.0
        ep_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                agent.clock.advance()

            # Action selection
            if goal_handcrafted:
                action_idx = _oracle_goal_action(
                    env, lambda_goal=lambda_goal, lambda_harm=lambda_harm,
                )
            else:
                action_idx = random.randint(0, action_dim - 1)

            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            oracle_prox_vals.append(_resource_proximity(env))

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")

            if ttype == "resource":
                ep_benefit += 1.0
            if ttype == "benefit_approach" and obs_body.dim() == 2 and obs_body.shape[-1] > 11:
                ep_benefit += float(obs_body[0, 11].item()) * 0.1

            if float(harm_signal) < 0:
                ep_harm += abs(float(harm_signal))

            if done:
                break

        benefit_per_ep.append(ep_benefit)
        harm_per_ep.append(ep_harm)

    avg_benefit = float(sum(benefit_per_ep) / max(1, len(benefit_per_ep)))
    avg_harm = float(sum(harm_per_ep) / max(1, len(harm_per_ep)))
    avg_oracle_prox = float(sum(oracle_prox_vals) / max(1, len(oracle_prox_vals)))

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" avg_benefit/ep={avg_benefit:.3f}"
        f" avg_harm/ep={avg_harm:.3f}"
        f" avg_oracle_prox={avg_oracle_prox:.3f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "goal_handcrafted": goal_handcrafted,
        "avg_benefit_per_ep": float(avg_benefit),
        "avg_harm_per_ep": float(avg_harm),
        "avg_oracle_prox": float(avg_oracle_prox),
        "train_resource_events": int(counts["resource"]),
    }


# ------------------------------------------------------------------ #
# Top-level run                                                        #
# ------------------------------------------------------------------ #

def run(
    seeds: Tuple = (42, 7, 13),
    warmup_episodes: int = 200,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    self_dim: int = 16,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.3,
    lambda_goal: float = 2.0,
    lambda_harm: float = 1.0,
    **kwargs,
) -> dict:
    results_handcrafted: List[Dict] = []
    results_absent:      List[Dict] = []

    for seed in seeds:
        for goal_handcrafted in [True, False]:
            label = "GOAL_HANDCRAFTED" if goal_handcrafted else "GOAL_ABSENT"
            print(
                f"\n[V3-EXQ-182a] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" lambda_goal={lambda_goal} lambda_harm={lambda_harm}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                goal_handcrafted=goal_handcrafted,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                lambda_goal=lambda_goal,
                lambda_harm=lambda_harm,
            )
            if goal_handcrafted:
                results_handcrafted.append(r)
            else:
                results_absent.append(r)

    # ---- Aggregation ----
    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    benefit_handcrafted = _avg(results_handcrafted, "avg_benefit_per_ep")
    benefit_absent      = _avg(results_absent,      "avg_benefit_per_ep")
    harm_handcrafted    = _avg(results_handcrafted, "avg_harm_per_ep")
    harm_absent         = _avg(results_absent,      "avg_harm_per_ep")

    benefit_ratio = (
        benefit_handcrafted / max(1e-6, benefit_absent)
        if benefit_absent > 1e-6 else 0.0
    )

    # Pass criteria
    c1_pass = benefit_ratio >= 1.3
    c2_pass = benefit_handcrafted > 0.5

    all_pass     = c1_pass and c2_pass
    criteria_met = sum([c1_pass, c2_pass])
    status       = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c2_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-182a] Final results:", flush=True)
    print(
        f"  benefit_handcrafted={benefit_handcrafted:.3f}"
        f"  benefit_absent={benefit_absent:.3f}"
        f"  ratio={benefit_ratio:.2f}x",
        flush=True,
    )
    print(
        f"  harm_handcrafted={harm_handcrafted:.3f}"
        f"  harm_absent={harm_absent:.3f}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/2)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.3x."
            " Oracle goal gradient does not produce 30% benefit improvement."
            " If ratio < 1.0: harm avoidance dominates even perfect goal signal"
            " -- lambda_goal/lambda_harm balance needs tuning, or env too lethal."
            " If 1.0 < ratio < 1.3: goal signal works but margin is small"
            " -- longer eval or more resources might help."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: benefit_handcrafted={benefit_handcrafted:.3f} <= 0.5."
            " Even a perfect oracle cannot collect meaningful resources."
            " Likely cause: harm avoidance dominates action selection;"
            " resource density too low; or env hazard layout blocks approach."
        )
    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_handcrafted_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" harm/ep={r['avg_harm_per_ep']:.3f}"
        f" oracle_prox={r['avg_oracle_prox']:.3f}"
        for r in results_handcrafted
    )
    per_absent_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" harm/ep={r['avg_harm_per_ep']:.3f}"
        for r in results_absent
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-182a -- SD-015 Handcrafted Goal Cue Diagnostic\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** SD-015\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n\n"
        f"## Design\n\n"
        f"Ceiling test: oracle goal signal (1/(1+manhattan_dist_to_resource))\n"
        f"computed directly from env state. No learned representation.\n"
        f"Action selection: score(a) = -lambda_goal*prox(a) + lambda_harm*hazard(a)\n"
        f"GOAL_ABSENT: random action selection.\n\n"
        f"**lambda_goal:** {lambda_goal}  **lambda_harm:** {lambda_harm}\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**Steps:** {steps_per_episode}/ep\n\n"
        f"## Results\n\n"
        f"| Condition | benefit/ep | harm/ep |\n"
        f"|---|---|---|\n"
        f"| GOAL_HANDCRAFTED | {benefit_handcrafted:.3f} | {harm_handcrafted:.3f} |\n"
        f"| GOAL_ABSENT | {benefit_absent:.3f} | {harm_absent:.3f} |\n\n"
        f"**Benefit ratio (handcrafted/absent): {benefit_ratio:.2f}x**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: benefit_ratio >= 1.3x | {'PASS' if c1_pass else 'FAIL'}"
        f" | {benefit_ratio:.2f}x |\n"
        f"| C2: benefit_handcrafted > 0.5 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {benefit_handcrafted:.3f} |\n\n"
        f"Criteria met: {criteria_met}/2 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"If PASS: The bottleneck in SD-015 goal navigation is in learned\n"
        f"goal representation (z_goal), not in the action-selection mechanism.\n"
        f"A perfect signal is sufficient -- z_goal learning must improve.\n\n"
        f"If C2 FAIL: Even a perfect goal signal is insufficient -- the\n"
        f"harm-avoidance tradeoff or env layout is the true bottleneck.\n"
        f"Improving z_goal learning alone will not fix SD-015.\n\n"
        f"## Per-Seed\n\n"
        f"GOAL_HANDCRAFTED:\n{per_handcrafted_rows}\n\n"
        f"GOAL_ABSENT:\n{per_absent_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "benefit_per_ep_handcrafted": float(benefit_handcrafted),
        "benefit_per_ep_absent":      float(benefit_absent),
        "benefit_ratio":              float(benefit_ratio),
        "harm_per_ep_handcrafted":    float(harm_handcrafted),
        "harm_per_ep_absent":         float(harm_absent),
        "lambda_goal":                float(lambda_goal),
        "lambda_harm":                float(lambda_harm),
        "n_seeds":                    float(len(seeds)),
        "alpha_world":                float(alpha_world),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    per_seed_results = []
    for i, seed in enumerate(seeds):
        per_seed_results.append({
            "seed": seed,
            "handcrafted": results_handcrafted[i],
            "absent": results_absent[i],
        })

    return {
        "status": status,
        "metrics": metrics,
        "per_seed_results": per_seed_results,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 1 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 7, 13])
    parser.add_argument("--warmup",          type=int,   default=200)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.3)
    parser.add_argument("--lambda-goal",     type=float, default=2.0)
    parser.add_argument("--lambda-harm",     type=float, default=1.0)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 1 seed, 2 warmup eps, 2 eval eps for smoke test")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds    = [42]
        args.warmup   = 2
        args.eval_eps = 2
        print("[DRY-RUN] 1 seed, 2 warmup, 2 eval", flush=True)

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        lambda_goal=args.lambda_goal,
        lambda_harm=args.lambda_harm,
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
