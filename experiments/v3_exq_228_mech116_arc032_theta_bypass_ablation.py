#!/opt/local/bin/python3
"""
V3-EXQ-228 -- MECH-116 / ARC-032: Proper Theta-Bypass Ablation

Claims: MECH-116, ARC-032
EXPERIMENT_PURPOSE = "evidence"

Scientific question: Does disabling the theta-buffer pathway (the mechanism
by which E1 goal-conditioned updates propagate to E3 via theta-averaged z_world)
specifically increase E1 prediction error on goal-relevant transitions compared
to goal-irrelevant transitions?

Context:
  EXQ-076f was non_contributory: it measured z_goal persistence (half-life after
  resource removal) but not E1 prediction quality under theta-bypass conditions.
  ARC-032's specific prediction: theta-bypass provides E1 goal-conditioned context
  that feeds E3 via theta_buffer.summary(). Without this pathway, E1 cannot
  contribute goal-relevant predictive context to E3's trajectory scoring.

  The theta-bypass pathway:
    1. E1._e1_tick(latent) is called each step.
    2. If goal_conditioned: E1 LSTM receives [z_self, z_world, z_goal] -> e1_prior.
    3. After E1 tick, theta_buffer.update(z_world, z_self) stores E1 estimates.
    4. E3 calls theta_buffer.summary() -> theta-averaged z_world passed to E3.
    5. E1's goal-conditioned processing shapes z_world predictions stored in buffer.
  Ablation: zero out z_theta (theta_buffer content) to prevent goal-conditioned
  E1 context from reaching E3. Specifically: after theta_buffer.update(), replace
  buffer entries with zeros for the THETA_BYPASS_DISABLED condition.

Design:
  THETA_BYPASS_ENABLED:  normal operation (theta_buffer feeds real z_world to E3)
  THETA_BYPASS_DISABLED: theta_buffer zeroed after each update (E3 sees zeros)
  Both conditions: z_goal_enabled=True, e1_goal_conditioned=True, drive_weight=2.0.
  Task: goal-directed, resource collection with hazards.
  3 seeds x 300 warmup + 300 eval eps x 200 steps.

  Measure E1 prediction error at eval time:
    - goal_zone: within GOAL_ZONE_RADIUS steps of nearest resource
    - neutral_zone: > NEUTRAL_RADIUS from ALL resources AND hazards
  Interaction effect:
    interaction = (err_neutral_disabled - err_neutral_enabled)
                - (err_goal_disabled    - err_goal_enabled)
  MECH-116: theta pathway supports E1 goal context for goal-relevant transitions.
  ARC-032:  theta pathway is necessary for goal maintenance (E1 prediction quality).

Pre-registered PASS criteria:
  C_MECH116: E1 prediction error is higher for goal-relevant transitions when
    theta-bypass disabled vs enabled: margin > 0.02 in >= 2/3 seeds.
    (err_goal_disabled - err_goal_enabled > 0.02)
  C_ARC032: interaction >= 0.01 in >= 2/3 seeds.
    (goal-specific degradation when theta-bypass disabled)

Precondition: z_goal_norm > 0.05 in THETA_BYPASS_ENABLED condition.
If not met: report substrate_limitation.

evidence_direction_per_claim:
  MECH-116: whether C_MECH116 passes
  ARC-032:  whether C_ARC032 passes (stronger selectivity test)
"""

import sys
import random
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_228_mech116_arc032_theta_bypass_ablation"
CLAIM_IDS = ["MECH-116", "ARC-032"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_GOAL_MARGIN   = 0.02   # C_MECH116: err_goal_disabled - err_goal_enabled
THRESH_INTERACTION   = 0.01   # C_ARC032:  interaction term
THRESH_GOAL_NORM     = 0.05   # Precondition: z_goal must seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_SIZE       = 10
WARMUP_EPISODES = 300
EVAL_EPISODES   = 300
STEPS_PER_EP    = 200
SEEDS           = [42, 7, 13]
WORLD_DIM       = 32
SELF_DIM        = 16
GOAL_ZONE_RADIUS = 3   # steps within this many grid cells of nearest resource
NEUTRAL_RADIUS   = 4   # steps this far from ALL resources AND hazards


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_toward_resource(env) -> int:
    ax, ay = env.agent_x, env.agent_y
    if not env.resources:
        return random.randint(0, env.action_dim - 1)
    best_d = float("inf")
    nearest = None
    for r in env.resources:
        rx, ry = int(r[0]), int(r[1])
        d = abs(ax - rx) + abs(ay - ry)
        if d < best_d:
            best_d = d
            nearest = (rx, ry)
    if nearest is None or best_d == 0:
        return random.randint(0, env.action_dim - 1)
    rx, ry = nearest
    dx, dy = rx - ax, ry - ay
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0
    else:
        return 3 if dy > 0 else 2


def _zone_classify(env, goal_zone_radius: int, neutral_radius: int):
    """Return (is_goal_zone, is_neutral) for current agent position."""
    ax = int(getattr(env, "agent_x", 0))
    ay = int(getattr(env, "agent_y", 0))

    d_to_resource = float("inf")
    if hasattr(env, "resources") and env.resources:
        for rx, ry in env.resources:
            d = abs(ax - int(rx)) + abs(ay - int(ry))
            if d < d_to_resource:
                d_to_resource = d

    d_to_hazard = float("inf")
    if hasattr(env, "hazards") and env.hazards:
        for hx, hy in env.hazards:
            d = abs(ax - int(hx)) + abs(ay - int(hy))
            if d < d_to_hazard:
                d_to_hazard = d

    is_goal_zone = d_to_resource <= goal_zone_radius
    is_neutral   = (d_to_resource > neutral_radius) and (d_to_hazard > neutral_radius)
    return is_goal_zone, is_neutral


# ---------------------------------------------------------------------------
# Run one condition (one seed)
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict:
    theta_disabled = (condition == "THETA_BYPASS_DISABLED")

    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.1,
        resource_respawn_on_consume=True,
    )
    n_actions = env.action_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=n_actions,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        z_goal_enabled=True,
        e1_goal_conditioned=True,
        goal_weight=1.0,
        benefit_eval_enabled=True,
        benefit_weight=0.5,
        drive_weight=2.0,
    )
    # Propagate goal_dim to E1 so goal_input_proj is active (EXQ-076f fix pattern)
    config.e1.goal_dim = WORLD_DIM

    agent = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-3)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-3)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()), lr=1e-3
    )
    harm_opt = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 4000

    def _zero_theta_buffer():
        """Zero all entries in agent.theta_buffer (THETA_BYPASS_DISABLED ablation)."""
        from collections import deque
        agent.theta_buffer._z_world_buffer = deque(
            [torch.zeros(1, WORLD_DIM, device=agent.device)
             for _ in agent.theta_buffer._z_world_buffer],
            maxlen=agent.theta_buffer.buffer_size,
        )
        agent.theta_buffer._z_self_buffer = deque(
            [torch.zeros(1, SELF_DIM, device=agent.device)
             for _ in agent.theta_buffer._z_self_buffer],
            maxlen=agent.theta_buffer.buffer_size,
        )

    agent.train()
    print(
        f"  [EXQ-228] {condition} seed={seed}"
        f" theta_disabled={theta_disabled}"
        f" e1_goal_dim={config.e1.goal_dim}"
        f" drive_weight={config.goal.drive_weight}",
        flush=True,
    )

    # ---- WARMUP ----
    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for step_i in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            # After E1 tick updates theta_buffer, optionally zero it
            agent._e1_tick(latent)
            if theta_disabled:
                _zero_theta_buffer()

            # 50% greedy to seed z_goal
            if random.random() < 0.5:
                action_idx = _greedy_toward_resource(env)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action_oh = _onehot(action_idx, n_actions, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            harm_val    = float(harm_signal)
            benefit_val = float(harm_signal) if harm_signal > 0 else 0.0

            # E1 + E2 training
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total.backward()
                e1_opt.step()
                e2_opt.step()

            # E3 harm training
            if agent._current_latent is not None:
                z_world_cur = agent._current_latent.z_world.detach()

                (harm_buf_pos if harm_val < 0 else harm_buf_neg).append(z_world_cur)
                if len(harm_buf_pos) > MAX_BUF: harm_buf_pos = harm_buf_pos[-MAX_BUF:]
                if len(harm_buf_neg) > MAX_BUF: harm_buf_neg = harm_buf_neg[-MAX_BUF:]

                if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                    k_p = min(16, len(harm_buf_pos))
                    k_n = min(16, len(harm_buf_neg))
                    pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                    ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                    zw_b = torch.cat(
                        [harm_buf_pos[i] for i in pi] +
                        [harm_buf_neg[i] for i in ni], dim=0
                    )
                    tgt = torch.cat([
                        torch.ones(k_p, 1, device=agent.device),
                        torch.zeros(k_n, 1, device=agent.device),
                    ], dim=0)
                    hl = F.binary_cross_entropy(agent.e3.harm_eval(zw_b), tgt)
                    if hl.requires_grad:
                        harm_opt.zero_grad()
                        hl.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_head.parameters(), 0.5
                        )
                        harm_opt.step()

            # z_goal update
            b_exp = 0.0
            if obs_body.dim() == 1 and obs_body.shape[0] > 11:
                b_exp = float(obs_body[11].item())
            elif obs_body.dim() > 1 and obs_body.shape[-1] > 11:
                b_exp = float(obs_body[0, 11].item())
            agent.update_z_goal(b_exp)

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            diag = agent.compute_goal_maintenance_diagnostic()
            print(
                f"  [warmup] {condition} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" goal_norm={diag['goal_norm']:.4f}",
                flush=True,
            )

    # Check precondition (THETA_BYPASS_ENABLED only)
    diag_final = agent.compute_goal_maintenance_diagnostic()
    goal_norm_f = float(diag_final["goal_norm"])
    precond_ok  = (condition == "THETA_BYPASS_DISABLED") or (goal_norm_f >= THRESH_GOAL_NORM)

    if not precond_ok:
        print(
            f"  [WARNING] {condition} seed={seed}"
            f" precond FAILED: goal_norm={goal_norm_f:.4f} < {THRESH_GOAL_NORM}"
            f" -- substrate_limitation",
            flush=True,
        )

    # ---- EVAL: collect E1 prediction error by zone ----
    agent.eval()

    err_goal_zone:    List[float] = []
    err_neutral_zone: List[float] = []

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                agent.clock.advance()
                agent._e1_tick(latent)
                if theta_disabled:
                    _zero_theta_buffer()

            # E1 prediction error (recompute outside no_grad for loss value)
            e1_loss_val = agent.compute_prediction_loss()
            e1_err = float(e1_loss_val.item()) if hasattr(e1_loss_val, "item") else 0.0

            # Zone classification
            is_goal_zone, is_neutral = _zone_classify(
                env, GOAL_ZONE_RADIUS, NEUTRAL_RADIUS
            )
            if is_goal_zone:
                err_goal_zone.append(e1_err)
            elif is_neutral:
                err_neutral_zone.append(e1_err)

            # Random action during eval
            action_idx = random.randint(0, n_actions - 1)
            action_oh  = _onehot(action_idx, n_actions, agent.device)
            agent._last_action = action_oh
            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            b_exp = 0.0
            if obs_body.dim() == 1 and obs_body.shape[0] > 11:
                b_exp = float(obs_body[11].item())
            elif obs_body.dim() > 1 and obs_body.shape[-1] > 11:
                b_exp = float(obs_body[0, 11].item())
            agent.update_z_goal(b_exp)

            if done:
                break

    def _mean(lst: List[float]) -> float:
        return float(sum(lst) / max(1, len(lst)))

    mean_err_goal    = _mean(err_goal_zone)
    mean_err_neutral = _mean(err_neutral_zone)

    print(
        f"  [{condition}] seed={seed}"
        f" n_goal={len(err_goal_zone)} n_neutral={len(err_neutral_zone)}"
        f" err_goal={mean_err_goal:.5f}"
        f" err_neutral={mean_err_neutral:.5f}"
        f" goal_norm={goal_norm_f:.4f}"
        f" precond_ok={precond_ok}",
        flush=True,
    )

    return {
        "seed":           seed,
        "condition":      condition,
        "mean_err_goal":  mean_err_goal,
        "mean_err_neutral": mean_err_neutral,
        "n_goal_steps":   len(err_goal_zone),
        "n_neutral_steps": len(err_neutral_zone),
        "goal_norm":      goal_norm_f,
        "precond_ok":     precond_ok,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    warmup = 2    if args.dry_run else WARMUP_EPISODES
    n_eval = 2    if args.dry_run else EVAL_EPISODES
    steps  = 20   if args.dry_run else STEPS_PER_EP
    seeds  = [42] if args.dry_run else SEEDS

    print(
        f"[V3-EXQ-228] MECH-116 / ARC-032 Theta-Bypass Ablation"
        f" dry_run={args.dry_run}",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        for condition in ["THETA_BYPASS_ENABLED", "THETA_BYPASS_DISABLED"]:
            res = _run_condition(
                seed=seed,
                condition=condition,
                warmup_episodes=warmup,
                eval_episodes=n_eval,
                steps_per_episode=steps,
            )
            all_results.append(res)

    # ---- Per-seed metrics ----
    per_seed: Dict[int, Dict] = {}
    for r in all_results:
        s = r["seed"]
        if s not in per_seed:
            per_seed[s] = {}
        per_seed[s][r["condition"]] = r

    seed_metrics = []
    for seed in seeds:
        if seed not in per_seed:
            continue
        conds = per_seed[seed]
        if "THETA_BYPASS_ENABLED" not in conds or "THETA_BYPASS_DISABLED" not in conds:
            continue
        en  = conds["THETA_BYPASS_ENABLED"]
        dis = conds["THETA_BYPASS_DISABLED"]

        goal_margin  = dis["mean_err_goal"] - en["mean_err_goal"]
        interaction  = (
            (dis["mean_err_neutral"] - en["mean_err_neutral"])
            - (dis["mean_err_goal"]  - en["mean_err_goal"])
        )

        c_mech116 = goal_margin >= THRESH_GOAL_MARGIN
        c_arc032  = interaction  >= THRESH_INTERACTION
        precond   = en["precond_ok"]

        print(
            f"  [EXQ-228] seed={seed}"
            f" goal_margin={goal_margin:.5f}"
            f" interaction={interaction:.5f}"
            f" C_MECH116={c_mech116} C_ARC032={c_arc032}"
            f" precond={precond}",
            flush=True,
        )

        seed_metrics.append({
            "seed":           seed,
            "goal_margin":    goal_margin,
            "interaction":    interaction,
            "err_goal_en":    en["mean_err_goal"],
            "err_goal_dis":   dis["mean_err_goal"],
            "err_neut_en":    en["mean_err_neutral"],
            "err_neut_dis":   dis["mean_err_neutral"],
            "goal_norm_en":   en["goal_norm"],
            "c_mech116":      c_mech116,
            "c_arc032":       c_arc032,
            "precond_ok":     precond,
        })

    n_seeds    = len(seed_metrics)
    precond_ok = all(m["precond_ok"] for m in seed_metrics)
    m116_count = sum(1 for m in seed_metrics if m["c_mech116"])
    arc032_count = sum(1 for m in seed_metrics if m["c_arc032"])

    c_mech116_pass = m116_count >= 2
    c_arc032_pass  = arc032_count >= 2

    if not precond_ok:
        outcome   = "FAIL"
        direction_mech116 = "non_contributory"
        direction_arc032  = "non_contributory"
        decision  = "substrate_limitation"
    elif c_mech116_pass and c_arc032_pass:
        outcome   = "PASS"
        direction_mech116 = "supports"
        direction_arc032  = "supports"
        decision  = "retain_ree"
    elif c_mech116_pass and not c_arc032_pass:
        outcome   = "PARTIAL"
        direction_mech116 = "supports"
        direction_arc032  = "does_not_support"
        decision  = "hybridize"
    else:
        outcome   = "FAIL"
        direction_mech116 = "does_not_support"
        direction_arc032  = "does_not_support"
        decision  = "retire_ree_claim"

    def _mean_sm(key: str) -> float:
        return sum(m[key] for m in seed_metrics) / max(1, len(seed_metrics))

    print(
        f"\n[V3-EXQ-228] RESULT: {outcome}"
        f" C_MECH116={c_mech116_pass} ({m116_count}/{n_seeds})"
        f" C_ARC032={c_arc032_pass} ({arc032_count}/{n_seeds})"
        f" precond_ok={precond_ok}",
        flush=True,
    )
    print(
        f"  mean_goal_margin={_mean_sm('goal_margin'):.5f}"
        f"  mean_interaction={_mean_sm('interaction'):.5f}"
        f"  mean_goal_norm_en={_mean_sm('goal_norm_en'):.4f}",
        flush=True,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    ts      = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    manifest = {
        "run_id":                f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":       EXPERIMENT_TYPE,
        "architecture_epoch":    "ree_hybrid_guardrails_v1",
        "claim_ids":             CLAIM_IDS,
        "experiment_purpose":    EXPERIMENT_PURPOSE,
        "outcome":               outcome,
        "evidence_direction":    direction_mech116,  # primary
        "evidence_direction_per_claim": {
            "MECH-116": direction_mech116,
            "ARC-032":  direction_arc032,
        },
        "decision":              decision,
        "timestamp":             ts,
        "seeds":                 seeds,
        "warmup_episodes":       warmup,
        "eval_episodes":         n_eval,
        "steps_per_episode":     steps,
        "world_dim":             WORLD_DIM,
        "drive_weight":          2.0,
        "thresh_goal_margin":    THRESH_GOAL_MARGIN,
        "thresh_interaction":    THRESH_INTERACTION,
        "thresh_goal_norm":      THRESH_GOAL_NORM,
        "goal_zone_radius":      GOAL_ZONE_RADIUS,
        "neutral_radius":        NEUTRAL_RADIUS,
        # Aggregate
        "mean_goal_margin":      _mean_sm("goal_margin"),
        "mean_interaction":      _mean_sm("interaction"),
        "mean_goal_norm_enabled": _mean_sm("goal_norm_en"),
        # Criteria
        "c_mech116_pass":        c_mech116_pass,
        "c_arc032_pass":         c_arc032_pass,
        "c_mech116_count":       m116_count,
        "c_arc032_count":        arc032_count,
        "precond_ok":            precond_ok,
        "n_seeds":               n_seeds,
        "seed_metrics":          seed_metrics,
        "all_condition_results": all_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-228] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
