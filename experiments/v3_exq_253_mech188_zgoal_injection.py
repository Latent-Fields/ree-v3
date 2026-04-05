#!/opt/local/bin/python3
"""
V3-EXQ-253 -- MECH-188: z_goal Injection (PFC Top-Down Goal Persistence)

Claims: MECH-188
EXPERIMENT_PURPOSE = "evidence"

=== SCIENTIFIC QUESTION ===

Does injecting a constant z_goal signal (simulating PFC top-down goal
persistence via DRN-mPFC serotonergic pathway, Miyazaki et al. 2020)
prevent behavioral flattening in LONG_HORIZON even when benefit terrain
has collapsed (z_goal seeding has failed)?

MECH-188 predicts: tonic PFC top-down goal maintenance (z_goal_inject)
can substitute for failed terrain-based seeding, preserving the
PLANNED/HABIT behavioral gap that collapses in the depression attractor.

This is distinct from:
  - MECH-186: serotonergic terrain maintenance (benefit terrain floor)
  - MECH-187: seeding gain regulation (incentive salience gain)
  - MECH-188 (this): top-down override of terrain-collapsed goal state

=== BIOLOGICAL GROUNDING ===

Miyazaki et al. (2020) showed DRN serotonin to mPFC specifically promotes
waiting (goal persistence) under temporal uncertainty -- the exact condition
when benefit terrain is temporarily collapsed (LONG_HORIZON). The z_goal
injection simulates this PFC top-down signal maintaining goal representation
when bottom-up seeding fails.

Maps to behavioural activation therapy: externally scaffolding goal structure
when depressive terrain collapse has removed the spontaneous seeding signal.

Key test: is MECH-188 sufficient in isolation to rescue behavioral gap?
INV-052 (EXP-0101) tests whether any single mechanism suffices for full
recovery; EXQ-253 tests whether MECH-188 alone maintains a meaningful gap.

=== DESIGN ===

Fixed LONG_HORIZON condition: 8x8, 1 resource, 3 hazards, 150 steps/ep,
hazard_harm=0.02 (same as EXQ-249 -- depression attractor confirmed context).

3 seeds, 2 conditions:
  (A) BASELINE: z_goal_inject=0.0 -- terrain-based seeding only (expected collapse)
  (B) INJECTED: z_goal_inject=0.3 -- constant norm floor on z_goal during action selection

Both conditions use PLANNED mode (z_goal_enabled=True, drive_weight=2) to
isolate the injection effect. Also run HABIT mode (z_goal_enabled=False) for
each condition to compute the behavioral gap.

Warmup: 200 eps; Eval: 100 eps.

Per seed x condition:
  - z_goal_norm (mean from warmup diagnostic)
  - PLANNED_resource_rate, HABIT_resource_rate
  - behavioral_gap = PLANNED_resource_rate - HABIT_resource_rate
  - harm_rate (PLANNED)

=== PRE-REGISTERED CRITERIA ===

PASS: Condition B vs A, majority (>=2/3 seeds):
  behavioral_gap(B) >= 0.05
  (injection maintains PLANNED advantage despite terrain collapse)

FAIL:
  < 2/3 seeds meeting gap criterion -> does_not_support
  (injection insufficient to maintain goal-directed behavior alone)

z_goal_norm(B) > z_goal_norm(A) is expected as confirmation that injection
is active (diagnostic, not pass criterion).
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
EXPERIMENT_TYPE    = "v3_exq_253_mech188_zgoal_injection"
CLAIM_IDS          = ["MECH-188"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
BEHAVIORAL_GAP_THRESH = 0.05   # PASS: gap(B) >= this (injection maintains PLANNED advantage)
PASS_SEED_COUNT       = 2      # need >= this many seeds meeting gap criterion (majority of 3)

# ---------------------------------------------------------------------------
# Grid and episode parameters (LONG_HORIZON)
# ---------------------------------------------------------------------------
GRID_SIZE    = 8
N_RESOURCES  = 1
N_HAZARDS    = 3
STEPS_PER_EP = 150
HAZARD_HARM  = 0.02

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

# z_goal injection magnitudes
INJECT_BASELINE  = 0.0   # Condition A: no injection
INJECT_TREATMENT = 0.3   # Condition B: constant 0.3 floor on z_goal norm


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
    """Manhattan distance from agent to nearest resource (999 if no resources)."""
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
    """Update z_goal from current step benefit_exposure and drive_level."""
    b_exp = _get_benefit_exposure(obs_body)
    energy = _get_energy(obs_body)
    drive_level = max(0.0, 1.0 - energy)
    agent.update_z_goal(b_exp, drive_level=drive_level)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    """Create LONG_HORIZON environment matching EXQ-237a / EXQ-249 parameters."""
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=N_RESOURCES,
        num_hazards=N_HAZARDS,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        hazard_harm=HAZARD_HARM,
        proximity_harm_scale=0.3,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        env_drift_interval=999,
        env_drift_prob=0.0,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(
    mode: str,
    condition: str,
    env: CausalGridWorldV2,
    seed: int,
) -> REEAgent:
    """
    Create agent for given mode (HABIT/PLANNED) and condition (A/B).

    mode: 'HABIT' (z_goal_enabled=False) or 'PLANNED' (z_goal_enabled=True)
    condition: 'A' (inject=0.0) or 'B' (inject=0.3)
    """
    planned = (mode == "PLANNED")
    inject  = INJECT_TREATMENT if condition == "B" else INJECT_BASELINE
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
        z_goal_enabled=planned,
        e1_goal_conditioned=planned,
        goal_weight=1.0 if planned else 0.0,
        drive_weight=2.0 if planned else 0.0,
        z_goal_inject=inject if planned else 0.0,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    mode: str,
    condition: str,
    warmup_episodes: int,
    seed: int,
) -> Dict:
    """
    Mixed-policy warmup: 40% greedy, 60% random.

    Trains E1, E2 world_forward, E3 harm_eval, E3 benefit_eval.
    PLANNED: additionally updates z_goal each step.

    Note: z_goal injection applies during select_action() (eval phase) only.
    During warmup we use mixed policy, so the injection's effect on trajectory
    scoring is minimal. The inject parameter is in the config and will take
    effect when select_action() is called in the eval phase.
    """
    planned = (mode == "PLANNED")
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

        for step_i in range(STEPS_PER_EP):
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

            # Mixed warmup policy
            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_toward_resource(env)
            else:
                action_idx = random.randint(0, n_act - 1)
            action_oh = _onehot(action_idx, n_act, device)
            agent._last_action = action_oh

            dist    = _dist_to_nearest_resource(env)
            is_near = 1.0 if dist <= 2 else 0.0

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            if planned:
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

            # Train E3 benefit_eval (proximity labels)
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
            inject_val = getattr(
                getattr(agent.config, "goal", None), "z_goal_inject", 0.0
            )
            print(
                f"    [train] cond={condition} mode={mode} seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
                f" harm_pos={len(harm_pos_buf)}"
                f" harm_neg={len(harm_neg_buf)}"
                f" goal_norm={diag['goal_norm']:.3f}"
                f" inject={inject_val:.2f}",
                flush=True,
            )

    diag_final = agent.compute_goal_maintenance_diagnostic()
    return {"goal_norm": float(diag_final["goal_norm"])}


# ---------------------------------------------------------------------------
# Eval (full agent pipeline)
# ---------------------------------------------------------------------------

def _eval(
    agent: REEAgent,
    env: CausalGridWorldV2,
    mode: str,
    condition: str,
    eval_episodes: int,
    seed: int,
) -> Dict:
    """
    Eval using the full agent pipeline: generate_trajectories + select_action.

    For PLANNED condition B, the z_goal injection (z_goal_inject=0.3) is
    active within select_action() via agent.goal_state.with_injection().
    This is the key test: does the injection produce a PLANNED/HABIT gap?
    """
    planned = (mode == "PLANNED")
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

        for _ in range(STEPS_PER_EP):
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
                action = _onehot(random.randint(0, n_act - 1), n_act, device)
                agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)

            if info.get("transition_type") == "resource":
                ep_resources += 1
            if float(harm_signal) < 0:
                ep_harm_sum += abs(float(harm_signal))
            ep_steps += 1

            if planned:
                with torch.no_grad():
                    _update_z_goal(agent, obs_dict["body_state"])

            if done:
                break

        resource_counts.append(1 if ep_resources >= 1 else 0)
        harm_rates.append(ep_harm_sum / max(1, ep_steps))

    resource_rate = sum(resource_counts) / max(1, len(resource_counts))
    harm_rate     = sum(harm_rates)      / max(1, len(harm_rates))

    return {
        "resource_rate": resource_rate,
        "harm_rate":     harm_rate,
    }


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def _run_seed(
    seed:            int,
    condition:       str,
    warmup_episodes: int,
    eval_episodes:   int,
) -> Dict:
    """Run HABIT and PLANNED modes for one seed in one condition."""
    results: Dict = {}

    for mode in ["HABIT", "PLANNED"]:
        print(
            f"\n[V3-EXQ-253] Seed {seed} cond={condition} mode={mode}"
            f" steps={STEPS_PER_EP}",
            flush=True,
        )
        env   = _make_env(seed)
        agent = _make_agent(mode, condition, env, seed)

        warmup_res = _warmup(agent, env, mode, condition, warmup_episodes, seed)
        eval_res   = _eval(agent, env, mode, condition, eval_episodes, seed)

        inject_used = getattr(
            getattr(agent.config, "goal", None), "z_goal_inject", 0.0
        )
        print(
            f"  [eval done] seed={seed} cond={condition} mode={mode}"
            f" resource_rate={eval_res['resource_rate']:.3f}"
            f" harm_rate={eval_res['harm_rate']:.5f}"
            f" goal_norm={warmup_res['goal_norm']:.3f}"
            f" inject={inject_used:.2f}",
            flush=True,
        )
        results[mode] = {
            "resource_rate": eval_res["resource_rate"],
            "harm_rate":     eval_res["harm_rate"],
            "goal_norm":     warmup_res["goal_norm"],
        }

    gap = results["PLANNED"]["resource_rate"] - results["HABIT"]["resource_rate"]
    gap_pass = gap >= BEHAVIORAL_GAP_THRESH
    print(
        f"  behavioral_gap={gap:.4f} (thresh>={BEHAVIORAL_GAP_THRESH})"
        f" pass={gap_pass}",
        flush=True,
    )
    print(
        f"verdict: {'PASS' if gap_pass else 'FAIL'} seed={seed} cond={condition}",
        flush=True,
    )

    return {
        "HABIT":           results["HABIT"],
        "PLANNED":         results["PLANNED"],
        "behavioral_gap":  gap,
        "gap_pass":        gap_pass,
    }


# ---------------------------------------------------------------------------
# Aggregate and criteria
# ---------------------------------------------------------------------------

def _aggregate(
    results_A: List[Dict],
    results_B: List[Dict],
    seeds: List[int],
) -> Dict:
    """
    Compute per-seed and condition comparison metrics.

    PASS criterion: Condition B vs A, majority (>=PASS_SEED_COUNT seeds):
      behavioral_gap(B) >= BEHAVIORAL_GAP_THRESH
    """
    per_seed_A = []
    per_seed_B = []

    for i, (ra, rb) in enumerate(zip(results_A, results_B)):
        per_seed_A.append({
            "seed":              seeds[i],
            "condition":         "A",
            "PLANNED_rr":        ra["PLANNED"]["resource_rate"],
            "HABIT_rr":          ra["HABIT"]["resource_rate"],
            "behavioral_gap":    ra["behavioral_gap"],
            "PLANNED_harm_rate": ra["PLANNED"]["harm_rate"],
            "z_goal_norm":       ra["PLANNED"]["goal_norm"],
            "gap_pass":          ra["gap_pass"],
        })
        per_seed_B.append({
            "seed":              seeds[i],
            "condition":         "B",
            "PLANNED_rr":        rb["PLANNED"]["resource_rate"],
            "HABIT_rr":          rb["HABIT"]["resource_rate"],
            "behavioral_gap":    rb["behavioral_gap"],
            "PLANNED_harm_rate": rb["PLANNED"]["harm_rate"],
            "z_goal_norm":       rb["PLANNED"]["goal_norm"],
            "gap_pass":          rb["gap_pass"],
        })

    def _mean(lst, key):
        vals = [s[key] for s in lst]
        return sum(vals) / max(1, len(vals))

    pass_count_B = sum(1 for s in per_seed_B if s["gap_pass"])
    overall_pass = pass_count_B >= PASS_SEED_COUNT

    return {
        "per_seed_A":     per_seed_A,
        "per_seed_B":     per_seed_B,
        "n_seeds":        len(seeds),
        "condition_comparison": {
            "mean_gap_A":        _mean(per_seed_A, "behavioral_gap"),
            "mean_gap_B":        _mean(per_seed_B, "behavioral_gap"),
            "mean_z_goal_norm_A": _mean(per_seed_A, "z_goal_norm"),
            "mean_z_goal_norm_B": _mean(per_seed_B, "z_goal_norm"),
            "mean_PLANNED_rr_A": _mean(per_seed_A, "PLANNED_rr"),
            "mean_PLANNED_rr_B": _mean(per_seed_B, "PLANNED_rr"),
            "mean_HABIT_rr_A":   _mean(per_seed_A, "HABIT_rr"),
            "mean_HABIT_rr_B":   _mean(per_seed_B, "HABIT_rr"),
            "mean_harm_rate_A":  _mean(per_seed_A, "PLANNED_harm_rate"),
            "mean_harm_rate_B":  _mean(per_seed_B, "PLANNED_harm_rate"),
        },
        "pass_count_B":   pass_count_B,
        "overall_pass":   overall_pass,
    }


def _decide(agg: Dict) -> Tuple[str, str, str]:
    """Return (outcome, evidence_direction, decision)."""
    if agg["overall_pass"]:
        return "PASS", "supports", "injection_maintains_gap"
    return "FAIL", "does_not_support", "injection_insufficient"


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
        f"[V3-EXQ-253] MECH-188: z_goal Injection (PFC top-down goal persistence)"
        f" dry_run={args.dry_run}",
        flush=True,
    )
    print(
        f"  LONG_HORIZON: {N_RESOURCES} resource, {N_HAZARDS} hazards,"
        f" {STEPS_PER_EP} steps, hazard_harm={HAZARD_HARM}",
        flush=True,
    )
    print(
        f"  Condition A: z_goal_inject={INJECT_BASELINE} (no injection, baseline)",
        flush=True,
    )
    print(
        f"  Condition B: z_goal_inject={INJECT_TREATMENT} (constant norm floor)",
        flush=True,
    )
    print(
        f"  PASS criterion: >={PASS_SEED_COUNT}/{len(seeds)} seeds with"
        f" gap(B)>={BEHAVIORAL_GAP_THRESH}",
        flush=True,
    )

    all_results_A: List[Dict] = []
    all_results_B: List[Dict] = []

    for seed in seeds:
        print(f"\n[V3-EXQ-253] === Seed {seed} ===", flush=True)

        print(f"\n[V3-EXQ-253] Seed {seed} -- Condition A (baseline)", flush=True)
        res_A = _run_seed(
            seed=seed, condition="A",
            warmup_episodes=warmup, eval_episodes=n_eval,
        )
        all_results_A.append(res_A)

        print(f"\n[V3-EXQ-253] Seed {seed} -- Condition B (injected)", flush=True)
        res_B = _run_seed(
            seed=seed, condition="B",
            warmup_episodes=warmup, eval_episodes=n_eval,
        )
        all_results_B.append(res_B)

    agg = _aggregate(all_results_A, all_results_B, seeds)
    cc  = agg["condition_comparison"]

    outcome, direction, decision = _decide(agg)

    print(f"\n[V3-EXQ-253] === Results ===", flush=True)
    print(
        f"  Condition A (no injection): mean_gap={cc['mean_gap_A']:.4f}"
        f"  z_goal_norm={cc['mean_z_goal_norm_A']:.3f}",
        flush=True,
    )
    print(
        f"  Condition B (injected 0.3): mean_gap={cc['mean_gap_B']:.4f}"
        f"  z_goal_norm={cc['mean_z_goal_norm_B']:.3f}",
        flush=True,
    )
    print(
        f"  Gap criterion pass count (B): {agg['pass_count_B']}/{agg['n_seeds']}"
        f" (need >={PASS_SEED_COUNT})",
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
        "steps_per_ep":       STEPS_PER_EP,
        "n_resources":        N_RESOURCES,
        "n_hazards":          N_HAZARDS,
        "hazard_harm":        HAZARD_HARM,
        "greedy_frac":        GREEDY_FRAC,
        "z_goal_inject_A":    INJECT_BASELINE,
        "z_goal_inject_B":    INJECT_TREATMENT,
        # Thresholds
        "behavioral_gap_thresh": BEHAVIORAL_GAP_THRESH,
        "pass_seed_count":       PASS_SEED_COUNT,
        # Condition comparison
        "condition_comparison": {k: float(v) for k, v in cc.items()},
        "pass_count_B":   agg["pass_count_B"],
        "overall_pass":   agg["overall_pass"],
        # Per-seed detail
        "per_seed_A": agg["per_seed_A"],
        "per_seed_B": agg["per_seed_B"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-253] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
