#!/opt/local/bin/python3
"""
V3-EXQ-252 -- MECH-187: Bidirectional z_goal Seeding Gain (5-condition design)

Claims: MECH-187
EXPERIMENT_PURPOSE = "evidence"

=== SCIENTIFIC QUESTION ===

Does z_goal seeding respond bidirectionally to gain manipulation?
Specifically:
  (P1) Moderate suppression (gain=0.6) collapses z_goal_norm in SIMPLE context
       where seeding would otherwise succeed (depression-analog).
  (P2) Moderate elevation (gain=2.0) rescues z_goal_norm in LONG_HORIZON context
       where terrain collapse makes seeding difficult without amplification.

Bidirectional confirmation is a strong discriminating test for MECH-187
(serotonergic gain regulation of incentive salience attribution).

=== DESIGN BASIS ===

Empirical anchor: Korte et al. 2016 (5-HT1B NAc microdialysis).
Empirically supported modulation range: ~x0.6-0.8 suppression, ~x1.5-2.5 elevation.
x0.2 and x5.0 included as outer sensitivity bounds (original EXP-0099 values).

=== 5-CONDITION DESIGN ===

All conditions: PLANNED mode (z_goal_enabled=True, drive_weight=2.0, benefit_eval=True).

(A) SIMPLE,        gain=1.0  -- healthy baseline
(B) SIMPLE,        gain=0.6  -- moderate suppression (empirical anchor)
(C) SIMPLE,        gain=0.2  -- strong suppression (outer sensitivity bound)
(D) LONG_HORIZON,  gain=2.0  -- moderate elevation (empirical anchor)
(E) LONG_HORIZON,  gain=5.0  -- strong elevation (outer sensitivity bound)

SIMPLE:       8x8, 2 resources, 1 hazard, 80 steps/ep  (easy -- seeding expected)
LONG_HORIZON: 8x8, 1 resource,  3 hazards, 150 steps/ep (hard -- seeding degraded)

=== PRE-REGISTERED CRITERIA ===

Evaluated per seed, majority rule (>= 2/3 seeds):

  P1: z_goal_norm(B) < 0.1 in SIMPLE  (moderate suppression collapses seeding)
  P2: z_goal_norm(D) >= 0.10 in LONG_HORIZON  (moderate elevation rescues seeding)
  PASS: P1 AND P2

Auxiliary (documentation only, not PASS/FAIL):
  A1: z_goal_norm(A) >= 0.10 in SIMPLE (baseline seeding healthy)
  A2: z_goal_norm(C) < z_goal_norm(B)  (strong suppression worse than moderate)
  A3: z_goal_norm(E) >= z_goal_norm(D) (strong elevation >= moderate elevation)

Evidence directions:
  PASS (P1 AND P2)     -> supports (MECH-187 bidirectional prediction confirmed)
  P1 PASS, P2 FAIL     -> does_not_support (suppression works, elevation doesn't)
  P1 FAIL, P2 PASS     -> does_not_support (elevation works, suppression doesn't)
  A1 FAIL              -> non_contributory (baseline seeding not established)
  Both fail            -> does_not_support
"""

import sys
import random
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
EXPERIMENT_TYPE    = "v3_exq_252_mech187_zgoal_seeding_gain"
CLAIM_IDS          = ["MECH-187"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
# P1: moderate suppression collapses seeding in SIMPLE
SUPPRESSION_THRESH  = 0.1    # z_goal_norm(B) must be < this
# P2: moderate elevation rescues seeding in LONG_HORIZON
ELEVATION_THRESH    = 0.10   # z_goal_norm(D) must be >= this
# A1: baseline seeding healthy in SIMPLE
BASELINE_THRESH     = 0.10   # z_goal_norm(A) should be >= this

MAJORITY_THRESH     = 2      # majority = >= 2 out of 3 seeds

# ---------------------------------------------------------------------------
# Grid and episode parameters (matching EXQ-237a)
# ---------------------------------------------------------------------------
GRID_SIZE          = 8
SIMPLE_N_RESOURCES = 2
SIMPLE_N_HAZARDS   = 1
SIMPLE_STEPS       = 80

LONG_N_RESOURCES   = 1
LONG_N_HAZARDS     = 3
LONG_STEPS         = 150

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
WARMUP_EPISODES = 200
EVAL_EPISODES   = 100
SEEDS           = [42, 7, 13]
GREEDY_FRAC     = 0.4
MAX_BUF         = 4000
WF_BUF_MAX      = 2000
WORLD_DIM       = 32
BATCH_SIZE      = 16

# Learning rates
LR_E1      = 1e-3
LR_E2_WF   = 1e-3
LR_HARM    = 1e-4
LR_BENEFIT = 1e-3

# ---------------------------------------------------------------------------
# Conditions: (label, context, gain)
# ---------------------------------------------------------------------------
CONDITIONS = [
    ("A", "SIMPLE",       1.0),
    ("B", "SIMPLE",       0.6),
    ("C", "SIMPLE",       0.2),
    ("D", "LONG_HORIZON", 2.0),
    ("E", "LONG_HORIZON", 5.0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_toward_resource(env) -> int:
    """Greedy action: move toward nearest resource (Manhattan)."""
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
    return 3 if dy > 0 else 2


def _dist_to_nearest_resource(env) -> int:
    if not env.resources:
        return 999
    ax, ay = env.agent_x, env.agent_y
    return min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)


def _get_benefit_exposure(obs_body: torch.Tensor) -> float:
    """Extract benefit_exposure from body_state obs (index 11 in proxy mode)."""
    flat = obs_body.flatten()
    if flat.shape[0] > 11:
        return float(flat[11].item())
    return 0.0


def _get_energy(obs_body: torch.Tensor) -> float:
    """Extract energy from body_state obs (index 3)."""
    flat = obs_body.flatten()
    if flat.shape[0] > 3:
        return float(flat[3].item())
    return 1.0


def _update_z_goal(agent: REEAgent, obs_body: torch.Tensor) -> None:
    """Update z_goal from current step's benefit_exposure and drive_level."""
    b_exp = _get_benefit_exposure(obs_body)
    energy = _get_energy(obs_body)
    drive_level = max(0.0, 1.0 - energy)
    agent.update_z_goal(b_exp, drive_level=drive_level)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(context: str, seed: int) -> CausalGridWorldV2:
    common = dict(
        seed=seed,
        size=GRID_SIZE,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        hazard_harm=0.02,
        proximity_harm_scale=0.3,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        env_drift_interval=999,
        env_drift_prob=0.0,
    )
    if context == "SIMPLE":
        return CausalGridWorldV2(
            num_resources=SIMPLE_N_RESOURCES,
            num_hazards=SIMPLE_N_HAZARDS,
            **common
        )
    else:  # LONG_HORIZON
        return CausalGridWorldV2(
            num_resources=LONG_N_RESOURCES,
            num_hazards=LONG_N_HAZARDS,
            **common
        )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(gain: float, env: CausalGridWorldV2, seed: int) -> REEAgent:
    """All conditions use PLANNED mode; gain varies."""
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=16,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=True,
        benefit_weight=0.5,
        z_goal_enabled=True,
        e1_goal_conditioned=True,
        goal_weight=1.0,
        drive_weight=2.0,
        z_goal_seeding_gain=gain,   # MECH-187: key experimental manipulation
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    label: str,
    warmup_episodes: int,
    steps_per_episode: int,
    seed: int,
) -> Dict:
    """
    Mixed-policy warmup: 40% greedy, 60% random.

    Trains: E1 (prediction loss), E2 world_forward,
    E3 harm_eval (stratified), E3 benefit_eval (proximity labels).
    z_goal updates each step (all conditions are PLANNED).
    """
    device  = agent.device
    n_act   = env.action_dim

    e1_params    = list(agent.e1.parameters())
    e2_wf_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    e1_opt      = optim.Adam(e1_params, lr=LR_E1)
    e2_wf_opt   = optim.Adam(e2_wf_params, lr=LR_E2_WF)
    harm_opt    = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_HARM)
    benefit_opt = optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=LR_BENEFIT)

    wf_buf:       List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_pos_buf: List[torch.Tensor] = []
    harm_neg_buf: List[torch.Tensor] = []
    ben_zw_buf:   List[torch.Tensor] = []
    ben_lbl_buf:  List[float]        = []

    random.seed(seed)
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for step_i in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            if ticks.get("e1_tick", False):
                _ = agent._e1_tick(latent)

            z_world_curr = latent.z_world.detach()

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev, action_prev, z_world_curr))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_toward_resource(env)
            else:
                action_idx = random.randint(0, n_act - 1)
            action_oh = _onehot(action_idx, n_act, device)
            agent._last_action = action_oh

            dist   = _dist_to_nearest_resource(env)
            is_near = 1.0 if dist <= 2 else 0.0

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            # z_goal update (all conditions)
            _update_z_goal(agent, obs_dict["body_state"])

            # Train E1
            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_opt.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
                    e1_opt.step()

            # Train E2 world_forward
            if len(wf_buf) >= BATCH_SIZE:
                idxs  = random.sample(range(len(wf_buf)), min(BATCH_SIZE, len(wf_buf)))
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_opt.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(e2_wf_params, 1.0)
                    e2_wf_opt.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            # Train E3 harm_eval (stratified)
            if float(harm_signal) < 0:
                harm_pos_buf.append(z_world_curr)
                if len(harm_pos_buf) > MAX_BUF:
                    harm_pos_buf = harm_pos_buf[-MAX_BUF:]
            else:
                harm_neg_buf.append(z_world_curr)
                if len(harm_neg_buf) > MAX_BUF:
                    harm_neg_buf = harm_neg_buf[-MAX_BUF:]

            if len(harm_pos_buf) >= 4 and len(harm_neg_buf) >= 4:
                k_p = min(BATCH_SIZE // 2, len(harm_pos_buf))
                k_n = min(BATCH_SIZE // 2, len(harm_neg_buf))
                pi  = torch.randperm(len(harm_pos_buf))[:k_p].tolist()
                ni  = torch.randperm(len(harm_neg_buf))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_pos_buf[i] for i in pi] + [harm_neg_buf[i] for i in ni],
                    dim=0,
                )
                tgt = torch.cat([
                    torch.ones(k_p,  1, device=device),
                    torch.zeros(k_n, 1, device=device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                hloss = F.binary_cross_entropy(pred, tgt)
                if hloss.requires_grad:
                    harm_opt.zero_grad()
                    hloss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_opt.step()

            # Train E3 benefit_eval
            ben_zw_buf.append(z_world_curr)
            ben_lbl_buf.append(is_near)
            if len(ben_zw_buf) > MAX_BUF:
                ben_zw_buf  = ben_zw_buf[-MAX_BUF:]
                ben_lbl_buf = ben_lbl_buf[-MAX_BUF:]

            if len(ben_zw_buf) >= 32 and step_i % 4 == 0:
                k    = min(32, len(ben_zw_buf))
                idxs = random.sample(range(len(ben_zw_buf)), k)
                zw_b = torch.cat([ben_zw_buf[i] for i in idxs], dim=0)
                lbl  = torch.tensor(
                    [ben_lbl_buf[i] for i in idxs],
                    dtype=torch.float32,
                ).unsqueeze(1).to(device)
                pred_b = agent.e3.benefit_eval(zw_b)
                bloss  = F.binary_cross_entropy(pred_b, lbl)
                if bloss.requires_grad:
                    benefit_opt.zero_grad()
                    bloss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.benefit_eval_head.parameters(), 0.5
                    )
                    benefit_opt.step()
                    agent.e3.record_benefit_sample(k)

            z_world_prev = z_world_curr
            action_prev  = action_oh

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            diag = agent.compute_goal_maintenance_diagnostic()
            print(
                f"    [train] cond={label}"
                f" seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
                f" harm_pos={len(harm_pos_buf)}"
                f" harm_neg={len(harm_neg_buf)}"
                f" goal_norm={diag['goal_norm']:.3f}",
                flush=True,
            )

    diag_final = agent.compute_goal_maintenance_diagnostic()
    return {"goal_norm": float(diag_final["goal_norm"])}


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def _eval(
    agent: REEAgent,
    env: CausalGridWorldV2,
    label: str,
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """Eval using full agent pipeline. z_goal updates continue (no gradients)."""
    agent.eval()

    device    = agent.device
    world_dim = WORLD_DIM
    n_act     = env.action_dim

    resource_counts: List[int]   = []
    harm_rates:      List[float] = []

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        ep_resources = 0
        ep_harm_sum  = 0.0
        ep_steps     = 0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            if ticks.get("e1_tick", False):
                e1_prior = agent._e1_tick(latent)
            else:
                e1_prior = torch.zeros(1, world_dim, device=device)

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks, temperature=1.0)

            if action is None:
                action = _onehot(
                    random.randint(0, n_act - 1), n_act, device
                )
                agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)

            if info.get("transition_type") == "resource":
                ep_resources += 1
            if float(harm_signal) < 0:
                ep_harm_sum += abs(float(harm_signal))
            ep_steps += 1

            with torch.no_grad():
                _update_z_goal(agent, obs_dict["body_state"])

            if done:
                break

        resource_counts.append(1 if ep_resources >= 1 else 0)
        harm_rates.append(ep_harm_sum / max(1, ep_steps))

    resource_rate = sum(resource_counts) / max(1, len(resource_counts))
    harm_rate     = sum(harm_rates)      / max(1, len(harm_rates))
    return {"resource_rate": resource_rate, "harm_rate": harm_rate}


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def _run_seed(
    seed:            int,
    warmup_episodes: int,
    eval_episodes:   int,
) -> Dict:
    """Run all 5 conditions for one seed."""
    results: Dict = {}

    for label, context, gain in CONDITIONS:
        steps = SIMPLE_STEPS if context == "SIMPLE" else LONG_STEPS
        print(
            f"\n[V3-EXQ-252] Seed {seed} cond={label} context={context}"
            f" gain={gain:.1f} steps={steps}",
            flush=True,
        )
        env   = _make_env(context, seed)
        agent = _make_agent(gain, env, seed)

        warmup_res = _warmup(
            agent, env, label, warmup_episodes, steps, seed,
        )
        eval_res = _eval(
            agent, env, label, eval_episodes, steps,
        )

        print(
            f"  [eval done] seed={seed} cond={label} ctx={context} gain={gain:.1f}"
            f" resource_rate={eval_res['resource_rate']:.3f}"
            f" harm_rate={eval_res['harm_rate']:.5f}"
            f" goal_norm={warmup_res['goal_norm']:.3f}",
            flush=True,
        )

        results[label] = {
            "context":       context,
            "gain":          gain,
            "resource_rate": eval_res["resource_rate"],
            "harm_rate":     eval_res["harm_rate"],
            "goal_norm":     warmup_res["goal_norm"],
        }

    return results


# ---------------------------------------------------------------------------
# Aggregate and criteria
# ---------------------------------------------------------------------------

def _aggregate(all_results: List[Dict]) -> Dict:
    n_seeds = len(all_results)

    per_seed = []
    for r in all_results:
        per_seed.append({
            cond: {
                "context":       r[cond]["context"],
                "gain":          r[cond]["gain"],
                "resource_rate": r[cond]["resource_rate"],
                "harm_rate":     r[cond]["harm_rate"],
                "goal_norm":     r[cond]["goal_norm"],
            }
            for cond in ["A", "B", "C", "D", "E"]
        })

    def _mean_field(cond, field):
        return sum(s[cond][field] for s in per_seed) / max(1, n_seeds)

    # P1: moderate suppression collapses seeding in SIMPLE (cond B)
    p1_flags = [s["B"]["goal_norm"] < SUPPRESSION_THRESH for s in per_seed]
    # P2: moderate elevation rescues seeding in LONG_HORIZON (cond D)
    p2_flags = [s["D"]["goal_norm"] >= ELEVATION_THRESH for s in per_seed]
    # A1: baseline seeding healthy in SIMPLE (cond A)
    a1_flags = [s["A"]["goal_norm"] >= BASELINE_THRESH for s in per_seed]
    # A2: strong suppression worse than moderate (cond C < B)
    a2_flags = [s["C"]["goal_norm"] < s["B"]["goal_norm"] for s in per_seed]
    # A3: strong elevation >= moderate elevation (cond E >= D)
    a3_flags = [s["E"]["goal_norm"] >= s["D"]["goal_norm"] for s in per_seed]

    p1_count = sum(p1_flags)
    p2_count = sum(p2_flags)
    a1_count = sum(a1_flags)
    a2_count = sum(a2_flags)
    a3_count = sum(a3_flags)

    p1_pass = p1_count >= MAJORITY_THRESH
    p2_pass = p2_count >= MAJORITY_THRESH
    a1_pass = a1_count >= MAJORITY_THRESH
    a2_pass = a2_count >= MAJORITY_THRESH
    a3_pass = a3_count >= MAJORITY_THRESH

    # Strength comparison per seed
    strength = []
    for s in per_seed:
        strength.append({
            "c_harder_than_b": s["C"]["goal_norm"] < s["B"]["goal_norm"],
            "e_rescues_more_than_d": s["E"]["goal_norm"] >= s["D"]["goal_norm"],
            "b_goal_norm": s["B"]["goal_norm"],
            "c_goal_norm": s["C"]["goal_norm"],
            "d_goal_norm": s["D"]["goal_norm"],
            "e_goal_norm": s["E"]["goal_norm"],
        })

    return {
        "per_seed":  per_seed,
        "n_seeds":   n_seeds,
        "means": {
            cond: {
                "goal_norm":     _mean_field(cond, "goal_norm"),
                "resource_rate": _mean_field(cond, "resource_rate"),
                "harm_rate":     _mean_field(cond, "harm_rate"),
            }
            for cond in ["A", "B", "C", "D", "E"]
        },
        "criteria": {
            "p1_suppression_simple": {"pass": p1_pass, "count": p1_count, "flags": p1_flags},
            "p2_elevation_long":     {"pass": p2_pass, "count": p2_count, "flags": p2_flags},
            "a1_baseline_healthy":   {"pass": a1_pass, "count": a1_count, "flags": a1_flags},
            "a2_c_worse_than_b":     {"pass": a2_pass, "count": a2_count, "flags": a2_flags},
            "a3_e_geq_d":            {"pass": a3_pass, "count": a3_count, "flags": a3_flags},
        },
        "bidirectional_confirmed": p1_pass and p2_pass,
        "strength_comparison": strength,
        "p1_pass": p1_pass,
        "p2_pass": p2_pass,
        "a1_pass": a1_pass,
        "a2_pass": a2_pass,
        "a3_pass": a3_pass,
    }


def _decide(agg: Dict) -> Tuple[str, str, str]:
    """Return (outcome, evidence_direction, decision)."""
    a1 = agg["a1_pass"]
    p1 = agg["p1_pass"]
    p2 = agg["p2_pass"]

    if not a1:
        return "FAIL", "non_contributory", "substrate_limitation_baseline_seeding_absent"
    if p1 and p2:
        return "PASS", "supports", "retain_ree_bidirectional_confirmed"
    if p1 and not p2:
        return "FAIL", "does_not_support", "elevation_failed_suppression_only"
    if not p1 and p2:
        return "FAIL", "does_not_support", "suppression_failed_elevation_only"
    return "FAIL", "does_not_support", "bidirectional_prediction_not_confirmed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    warmup = 5    if args.dry_run else WARMUP_EPISODES
    n_eval = 5    if args.dry_run else EVAL_EPISODES
    seeds  = [42] if args.dry_run else SEEDS

    print(
        f"[V3-EXQ-252] MECH-187: Bidirectional z_goal Seeding Gain"
        f" dry_run={args.dry_run}"
        f" warmup={warmup} eval={n_eval} seeds={seeds}",
        flush=True,
    )
    print(
        f"  5 conditions: A(SIMPLE/x1.0), B(SIMPLE/x0.6), C(SIMPLE/x0.2),"
        f" D(LONG/x2.0), E(LONG/x5.0)",
        flush=True,
    )
    print(
        f"  SIMPLE: {SIMPLE_N_RESOURCES} res, {SIMPLE_N_HAZARDS} haz, {SIMPLE_STEPS} steps",
        flush=True,
    )
    print(
        f"  LONG_HORIZON: {LONG_N_RESOURCES} res, {LONG_N_HAZARDS} haz, {LONG_STEPS} steps",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        print(f"\n[V3-EXQ-252] === Seed {seed} ===", flush=True)
        seed_results = _run_seed(
            seed=seed,
            warmup_episodes=warmup,
            eval_episodes=n_eval,
        )
        all_results.append(seed_results)

    agg = _aggregate(all_results)
    m   = agg["means"]
    cr  = agg["criteria"]

    outcome, direction, decision = _decide(agg)

    print(f"\n[V3-EXQ-252] === Results ===", flush=True)
    for cond_label, context, gain in CONDITIONS:
        print(
            f"  Cond {cond_label} ({context}/x{gain:.1f}):"
            f" goal_norm={m[cond_label]['goal_norm']:.3f}"
            f" resource_rate={m[cond_label]['resource_rate']:.3f}"
            f" harm_rate={m[cond_label]['harm_rate']:.5f}",
            flush=True,
        )
    print(
        f"  P1 (B goal_norm < {SUPPRESSION_THRESH}): {'PASS' if agg['p1_pass'] else 'FAIL'}"
        f" ({cr['p1_suppression_simple']['count']}/{agg['n_seeds']})",
        flush=True,
    )
    print(
        f"  P2 (D goal_norm >= {ELEVATION_THRESH}): {'PASS' if agg['p2_pass'] else 'FAIL'}"
        f" ({cr['p2_elevation_long']['count']}/{agg['n_seeds']})",
        flush=True,
    )
    print(
        f"  A1 (A goal_norm >= {BASELINE_THRESH})[info]: {'PASS' if agg['a1_pass'] else 'FAIL'}"
        f" ({cr['a1_baseline_healthy']['count']}/{agg['n_seeds']})",
        flush=True,
    )
    print(
        f"  A2 (C < B goal_norm)[info]: {'PASS' if agg['a2_pass'] else 'FAIL'}"
        f" ({cr['a2_c_worse_than_b']['count']}/{agg['n_seeds']})",
        flush=True,
    )
    print(
        f"  A3 (E >= D goal_norm)[info]: {'PASS' if agg['a3_pass'] else 'FAIL'}"
        f" ({cr['a3_e_geq_d']['count']}/{agg['n_seeds']})",
        flush=True,
    )
    print(
        f"  bidirectional_confirmed={agg['bidirectional_confirmed']}",
        flush=True,
    )
    print(
        f"  -> {outcome} decision={decision} direction={direction}",
        flush=True,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    # Write output
    ts      = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    manifest = {
        "run_id":             f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":    EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome":            outcome,
        "evidence_direction": direction,
        "decision":           decision,
        "timestamp":          ts,
        "seeds":              seeds,
        # Parameters
        "warmup_episodes":    warmup,
        "eval_episodes":      n_eval,
        "simple_steps":       SIMPLE_STEPS,
        "long_steps":         LONG_STEPS,
        "simple_n_resources": SIMPLE_N_RESOURCES,
        "simple_n_hazards":   SIMPLE_N_HAZARDS,
        "long_n_resources":   LONG_N_RESOURCES,
        "long_n_hazards":     LONG_N_HAZARDS,
        "greedy_frac":        GREEDY_FRAC,
        "hazard_harm":        0.02,
        "conditions": [
            {"label": lbl, "context": ctx, "gain": g}
            for lbl, ctx, g in CONDITIONS
        ],
        # Thresholds
        "suppression_thresh": SUPPRESSION_THRESH,
        "elevation_thresh":   ELEVATION_THRESH,
        "baseline_thresh":    BASELINE_THRESH,
        # Mean metrics per condition
        "means": {
            cond: m[cond]
            for cond in ["A", "B", "C", "D", "E"]
        },
        # Criteria
        "p1_suppression_pass": agg["p1_pass"],
        "p2_elevation_pass":   agg["p2_pass"],
        "a1_baseline_pass":    agg["a1_pass"],
        "a2_c_worse_than_b":   agg["a2_pass"],
        "a3_e_geq_d":          agg["a3_pass"],
        "bidirectional_confirmed": agg["bidirectional_confirmed"],
        "strength_comparison": agg["strength_comparison"],
        # Per-seed detail
        "per_seed": [
            {
                cond: {
                    "goal_norm":     s[cond]["goal_norm"],
                    "resource_rate": s[cond]["resource_rate"],
                    "harm_rate":     s[cond]["harm_rate"],
                }
                for cond in ["A", "B", "C", "D", "E"]
            }
            for s in agg["per_seed"]
        ],
        # Criteria detail
        "criteria": {
            k: {
                "pass": v["pass"],
                "count": v["count"],
                "flags": [bool(f) for f in v["flags"]],
            }
            for k, v in agg["criteria"].items()
        },
    }

    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[V3-EXQ-252] Output written: {out_path}", flush=True)
    print(f"verdict: {'PASS' if outcome == 'PASS' else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
