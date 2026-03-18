"""
V3-EXQ-013 — SD-005 Strict: Event-Conditional Separation

Claim: SD-005 (z_self/z_world split) produces functionally distinct channels
when evaluated under event-conditional conditions.

Motivation (2026-03-17):
  EXQ-001 used permissive aggregate-correlation tests. body_selectivity_margin
  was -0.137 (z_world was MORE sensitive to body movement than z_self — backwards)
  but this was not a PASS criterion. The split exists architecturally but is
  not functionally working. This experiment uses event-conditional measurement
  to properly test functional separation.

Three event types (collected separately):
  1. empty_move:          transition_type == "none". Perspective shift only.
                          z_self should change (body moved); z_world should
                          change mainly due to view scrolling (perspective shift).
  2. env_caused_hazard:   transition_type == "env_caused_hazard". Agent moved
                          into a static hazard cell. Genuine world-state change
                          (hazard entity appears at view center). z_world should
                          change more than empty_move; z_self changes similar to
                          any movement step.
  3. agent_caused_hazard: transition_type == "agent_caused_hazard". Agent moved
                          into contaminated terrain. Both z_self and z_world change.

PASS criteria (ALL must hold):
  C1: Δz_world(env_caused) > Δz_world(empty_move) with margin > 0.005
      World channel responds more to genuine world events than to locomotion.
  C2: Δz_self(empty_move) > Δz_world(empty_move)
      Body channel more responsive to locomotion than world channel.
      (This FAILED in EXQ-001 where margin was -0.137.)
  C3: E2 action-conditional divergence at near-hazard > at safe (margin > 0.002)
      ||E2.world_forward(z_world, a1) - E2.world_forward(z_world, a2)|| should
      be larger when agent is adjacent to a hazard than in open space.
  C4: n >= 30 for each event type (sufficient samples)
  C5: No fatal errors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_013_sd005_strict"
CLAIM_IDS = ["SD-005"]

E2_ROLLOUT_STEPS = 5
RECON_WEIGHT = 1.0


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _make_world_decoder(world_dim: int, world_obs_dim: int, hidden_dim: int = 64):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(world_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, world_obs_dim),
    )


def _train_episodes(
    agent: REEAgent,
    env: CausalGridWorld,
    world_decoder,
    optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """Training phase: E1 + E2_self + E2_world multi-step + reconstruction."""
    import torch.nn as nn
    agent.train()
    world_decoder.train()

    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200
    total_harm = 0
    total_benefit = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        episode_traj: List = []

        for step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            episode_traj.append((latent.z_world.detach(), action.detach()))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if harm_signal < 0:
                total_harm += 1
            elif harm_signal > 0:
                total_benefit += 1

            # E1 + E2_self loss
            e1_loss = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()

            # E2_world multi-step loss
            e2w_loss = _compute_multistep_e2_loss(agent, traj_buffer, E2_ROLLOUT_STEPS)

            # Reconstruction loss (anchors world encoder to obs)
            obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
            z_world_recon = agent.latent_stack.split_encoder.world_encoder(obs_w)
            recon = world_decoder(z_world_recon)
            recon_loss = F.mse_loss(recon, obs_w)

            total_loss = e1_loss + e2_self_loss + e2w_loss + RECON_WEIGHT * recon_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(world_decoder.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        for start in range(0, len(episode_traj) - E2_ROLLOUT_STEPS):
            traj_buffer.append(episode_traj[start:start + E2_ROLLOUT_STEPS + 1])
        if len(traj_buffer) > MAX_TRAJ_BUFFER:
            traj_buffer = traj_buffer[-MAX_TRAJ_BUFFER:]

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}  benefit={total_benefit}",
                  flush=True)

    return {"total_harm": total_harm, "total_benefit": total_benefit}


def _compute_multistep_e2_loss(agent, traj_buffer, rollout_steps, batch_size=8):
    if len(traj_buffer) < 2:
        return next(agent.e1.parameters()).sum() * 0.0
    n = min(batch_size, len(traj_buffer))
    idxs = torch.randperm(len(traj_buffer))[:n].tolist()
    total = next(agent.e1.parameters()).sum() * 0.0
    count = 0
    for idx in idxs:
        segment = traj_buffer[idx]
        if len(segment) < rollout_steps + 1:
            continue
        z_start = segment[0][0]
        z_target = segment[rollout_steps][0]
        z = z_start
        for k in range(rollout_steps):
            z = agent.e2.world_forward(z, segment[k][1])
        total = total + F.mse_loss(z, z_target)
        count += 1
    return total / count if count > 0 else total


def _collect_events(
    agent: REEAgent,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
) -> Tuple[Dict[str, List], List[float], List[float]]:
    """
    Collect per-step delta metrics categorized by event type.
    Also collect E2 action-conditional divergence at near-hazard vs safe positions.
    """
    agent.eval()

    # Event buckets: (delta_z_self, delta_z_world)
    events: Dict[str, List[Tuple[float, float]]] = {
        "none": [],
        "env_caused_hazard": [],
        "agent_caused_hazard": [],
    }

    near_hazard_divergences: List[float] = []
    safe_divergences: List[float] = []

    hazard_type = env.ENTITY_TYPES["hazard"]
    wall_type = env.ENTITY_TYPES["wall"]

    for episode in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_self_prev = None
        z_world_prev = None

        for step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                z_self_curr = latent.z_self.detach()
                z_world_curr = latent.z_world.detach()

                # Record deltas if we have a previous state
                action_idx = random.randint(0, env.action_dim - 1)
                action = _action_to_onehot(action_idx, env.action_dim, agent.device)
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info["transition_type"]

            # Record event deltas (after step, compare pre/post)
            if z_self_prev is not None:
                dz_self = float(torch.norm(z_self_curr - z_self_prev).item())
                dz_world = float(torch.norm(z_world_curr - z_world_prev).item())
                if ttype in events:
                    events[ttype].append((dz_self, dz_world))

            z_self_prev = z_self_curr
            z_world_prev = z_world_curr

            # E2 divergence test: evaluate at current position if interesting
            with torch.no_grad():
                ax, ay = env.agent_x, env.agent_y
                min_dist_to_hazard = min(
                    abs(ax - hx) + abs(ay - hy) for hx, hy in env.hazards
                ) if env.hazards else 999

                # Near-hazard probe: agent is adjacent (dist 1) to a hazard
                # Safe probe: agent is far (dist > 3) from any hazard
                # Only collect at non-wall, non-hazard positions
                cell = int(env.grid[ax, ay])
                if cell not in (wall_type, hazard_type):
                    # Pick two different actions
                    a1 = _action_to_onehot(0, env.action_dim, agent.device)  # up
                    a2 = _action_to_onehot(1, env.action_dim, agent.device)  # down

                    zw = latent.z_world
                    zw_a1 = agent.e2.world_forward(zw, a1)
                    zw_a2 = agent.e2.world_forward(zw, a2)
                    divergence = float(torch.norm(zw_a1 - zw_a2).item())

                    if min_dist_to_hazard == 1:
                        near_hazard_divergences.append(divergence)
                    elif min_dist_to_hazard > 3:
                        safe_divergences.append(divergence)

            if done:
                break

    return events, near_hazard_divergences, safe_divergences


def run(
    seed: int = 0,
    num_train_episodes: int = 1000,
    num_eval_episodes: int = 100,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    harm_scale: float = 0.02,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    # Dense hazards + env drift for sufficient env_caused events
    env = CausalGridWorld(
        seed=seed,
        size=12,
        num_hazards=15,
        num_resources=5,
        env_drift_interval=3,
        env_drift_prob=0.5,
        hazard_harm=harm_scale,
        contaminated_harm=harm_scale,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
    )
    agent = REEAgent(config)

    world_decoder = _make_world_decoder(world_dim, env.world_obs_dim)
    params = list(agent.parameters()) + list(world_decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    print(f"[V3-EXQ-013] Training {num_train_episodes} eps, 12x12 grid, 15 hazards, "
          f"env_drift_interval=3, env_drift_prob=0.5", flush=True)
    train_metrics = _train_episodes(
        agent, env, world_decoder, optimizer, num_train_episodes, steps_per_episode
    )

    print(f"[V3-EXQ-013] Collecting events ({num_eval_episodes} eval eps)...", flush=True)
    events, near_hazard_divs, safe_divs = _collect_events(
        agent, env, num_eval_episodes, steps_per_episode
    )

    # ── Analysis ────────────────────────────────────────────────────────
    def _event_stats(bucket: List[Tuple[float, float]]) -> Dict:
        if not bucket:
            return {"n": 0, "mean_dz_self": 0.0, "mean_dz_world": 0.0,
                    "std_dz_self": 0.0, "std_dz_world": 0.0}
        dz_self  = [x[0] for x in bucket]
        dz_world = [x[1] for x in bucket]
        return {
            "n": len(bucket),
            "mean_dz_self":  float(sum(dz_self)  / len(dz_self)),
            "mean_dz_world": float(sum(dz_world) / len(dz_world)),
            "std_dz_self":   float(torch.tensor(dz_self).std().item()),
            "std_dz_world":  float(torch.tensor(dz_world).std().item()),
        }

    stats_empty    = _event_stats(events["none"])
    stats_env      = _event_stats(events["env_caused_hazard"])
    stats_agent    = _event_stats(events["agent_caused_hazard"])

    mean_near_div = float(sum(near_hazard_divs) / max(1, len(near_hazard_divs)))
    mean_safe_div = float(sum(safe_divs)        / max(1, len(safe_divs)))
    e2_divergence_margin = mean_near_div - mean_safe_div

    # ── PASS / FAIL ──────────────────────────────────────────────────────
    # C1: z_world(env_caused) > z_world(empty_move) margin > 0.005
    c1_margin = stats_env["mean_dz_world"] - stats_empty["mean_dz_world"]
    c1_pass = c1_margin > 0.005

    # C2: z_self(empty_move) > z_world(empty_move) — body selectivity
    c2_margin = stats_empty["mean_dz_self"] - stats_empty["mean_dz_world"]
    c2_pass = c2_margin > 0.0

    # C3: E2 action-conditional divergence at near-hazard > safe (margin > 0.002)
    c3_pass = e2_divergence_margin > 0.002

    # C4: min n >= 30 for each event type
    n_empty = stats_empty["n"]
    n_env   = stats_env["n"]
    n_agent = stats_agent["n"]
    c4_pass = n_empty >= 30 and n_env >= 30 and n_agent >= 30

    # C5: no fatal errors (we got here, so PASS)
    c5_pass = True
    fatal_errors = 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: Δz_world(env_hazard)={stats_env['mean_dz_world']:.4f} vs "
            f"Δz_world(empty)={stats_empty['mean_dz_world']:.4f}  margin={c1_margin:.4f} <= 0.005"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: body_selectivity margin={c2_margin:.4f} <= 0.0 "
            f"[Δz_self(empty)={stats_empty['mean_dz_self']:.4f} "
            f"Δz_world(empty)={stats_empty['mean_dz_world']:.4f}]"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: E2 divergence margin={e2_divergence_margin:.4f} <= 0.002 "
            f"[near={mean_near_div:.4f} safe={mean_safe_div:.4f}]"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: event counts too low — "
            f"empty={n_empty} env={n_env} agent_caused={n_agent} (need >= 30 each)"
        )

    print(f"\nV3-EXQ-013 verdict: {status}  ({criteria_met}/5)", flush=True)
    print(f"  Events — empty_move: {n_empty}  env_hazard: {n_env}  agent_hazard: {n_agent}",
          flush=True)
    print(f"  C1: Δz_world env={stats_env['mean_dz_world']:.4f} "
          f"vs empty={stats_empty['mean_dz_world']:.4f} margin={c1_margin:.4f} "
          f"({'PASS' if c1_pass else 'FAIL'})", flush=True)
    print(f"  C2: body_sel margin={c2_margin:.4f} "
          f"[dz_self={stats_empty['mean_dz_self']:.4f} "
          f"dz_world={stats_empty['mean_dz_world']:.4f}] "
          f"({'PASS' if c2_pass else 'FAIL'})", flush=True)
    print(f"  C3: E2 div near={mean_near_div:.4f} safe={mean_safe_div:.4f} "
          f"margin={e2_divergence_margin:.4f} ({'PASS' if c3_pass else 'FAIL'})", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "fatal_error_count":          float(fatal_errors),
        "n_empty_moves":              float(n_empty),
        "n_env_caused_hazard":        float(n_env),
        "n_agent_caused_hazard":      float(n_agent),
        # Empty move stats
        "mean_dz_self_empty":         float(stats_empty["mean_dz_self"]),
        "mean_dz_world_empty":        float(stats_empty["mean_dz_world"]),
        "std_dz_self_empty":          float(stats_empty["std_dz_self"]),
        "std_dz_world_empty":         float(stats_empty["std_dz_world"]),
        # Env-caused hazard stats
        "mean_dz_self_env_hazard":    float(stats_env["mean_dz_self"]),
        "mean_dz_world_env_hazard":   float(stats_env["mean_dz_world"]),
        # Agent-caused hazard stats
        "mean_dz_self_agent_hazard":  float(stats_agent["mean_dz_self"]),
        "mean_dz_world_agent_hazard": float(stats_agent["mean_dz_world"]),
        # Criterion metrics
        "c1_world_margin":            float(c1_margin),
        "c2_body_selectivity_margin": float(c2_margin),
        "c3_e2_divergence_margin":    float(e2_divergence_margin),
        "mean_near_hazard_divergence": float(mean_near_div),
        "mean_safe_divergence":       float(mean_safe_div),
        "n_near_hazard_probe_steps":  float(len(near_hazard_divs)),
        "n_safe_probe_steps":         float(len(safe_divs)),
        # Training metrics
        "warmup_harm_events":         float(train_metrics["total_harm"]),
        "warmup_benefit_events":      float(train_metrics["total_benefit"]),
        "num_train_episodes":         float(num_train_episodes),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-013 — SD-005 Strict: Event-Conditional Separation

**Status:** {status}
**Training:** {num_train_episodes} eps (RANDOM policy, 12×12, 15 hazards, drift_interval=3, drift_prob=0.5)
**Eval:** {num_eval_episodes} eps
**Seed:** {seed}

## Motivation

EXQ-001 (SD-005 PASS) used aggregate correlations. body_selectivity_margin was −0.137
(z_world MORE sensitive to body movement than z_self — backwards) but was not a PASS criterion.
This experiment uses event-conditional measurement to test functional separation.

## Event Counts

| Event Type | n |
|---|---|
| empty_move (transition_type=none) | {n_empty} |
| env_caused_hazard | {n_env} |
| agent_caused_hazard | {n_agent} |

## Δz Statistics by Event Type

| Event Type | Δz_self (mean±std) | Δz_world (mean±std) |
|---|---|---|
| empty_move    | {stats_empty["mean_dz_self"]:.4f} ± {stats_empty["std_dz_self"]:.4f} | {stats_empty["mean_dz_world"]:.4f} ± {stats_empty["std_dz_world"]:.4f} |
| env_hazard    | {stats_env["mean_dz_self"]:.4f} ± {stats_env["std_dz_self"]:.4f} | {stats_env["mean_dz_world"]:.4f} ± {stats_env["std_dz_world"]:.4f} |
| agent_hazard  | {stats_agent["mean_dz_self"]:.4f} ± {stats_agent["std_dz_self"]:.4f} | {stats_agent["mean_dz_world"]:.4f} ± {stats_agent["std_dz_world"]:.4f} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: Δz_world(env_caused) > Δz_world(empty_move), margin > 0.005 | {"PASS" if c1_pass else "FAIL"} | margin={c1_margin:.4f} |
| C2: Δz_self(empty_move) > Δz_world(empty_move) (body selectivity) | {"PASS" if c2_pass else "FAIL"} | margin={c2_margin:.4f} |
| C3: E2 divergence near-hazard > safe, margin > 0.002 | {"PASS" if c3_pass else "FAIL"} | margin={e2_divergence_margin:.4f} |
| C4: n >= 30 per event type | {"PASS" if c4_pass else "FAIL"} | empty={n_empty} env={n_env} agent={n_agent} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | 0 |

## E2 Divergence Results

- Mean divergence at near-hazard positions: {mean_near_div:.4f}
- Mean divergence at safe positions: {mean_safe_div:.4f}
- Margin: {e2_divergence_margin:.4f}
- n_near_hazard probes: {len(near_hazard_divs)}, n_safe probes: {len(safe_divs)}

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens"),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": fatal_errors,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--train-episodes", type=int,   default=1000)
    parser.add_argument("--eval-episodes",  type=int,   default=100)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--harm-scale",     type=float, default=0.02)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        num_train_episodes=args.train_episodes,
        num_eval_episodes=args.eval_episodes,
        steps_per_episode=args.steps,
        harm_scale=args.harm_scale,
        alpha_world=args.alpha_world,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
