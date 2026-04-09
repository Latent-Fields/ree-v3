#!/opt/local/bin/python3
"""
V3-EXQ-268 -- INV-053: Multi-Seed Replication of Computational Depression Attractor

Claims: INV-053
EXPERIMENT_PURPOSE = "evidence"

=== SCIENTIFIC QUESTION ===

Is the computational depression attractor observed in EXQ-237a seed=42 robust
across seeds? Specifically: in the LONG_HORIZON condition (8x8, 1 resource,
3 hazards, 150 steps/ep, hazard_harm=0.02), do multiple seeds show:

  (1) z_goal seeding failure (z_goal_norm < 0.1 -- goal representation collapses)
  (2) Behavioral equivalence of HABIT and PLANNED modes based on harm_rate
      (|HABIT_harm_rate - PLANNED_harm_rate| < 0.02 -- modes indistinguishable)

EXQ-237a seed=42 provided preliminary evidence. This experiment provides the
systematic replication needed to register INV-053 as active.
Run before MECH-186/187/188 experiments (see EXP-0096 notes).

=== BIOLOGICAL GROUNDING ===

This experiment operationalises the "computational depression attractor" as:
  - z_goal_norm < 0.1: wanting collapse (VTA/hippocampal goal representation fails to seed)
  - |HABIT_harm_rate - PLANNED_harm_rate| < 0.02: HABIT == PLANNED (planning advantage
    disappears when goal pathway that differentiates them has failed)

Depression attractor = state where z_goal fails to seed because benefit terrain has
collapsed (hazard_harm overwhelms benefit signal -> drive cycles cannot bootstrap goal
representation). Once entered, both HABIT and PLANNED systems produce indistinguishable
harm rates because the goal pathway is non-functional.

=== DESIGN ===

Fixed LONG_HORIZON condition: same parameters as EXQ-237a LONG_HORIZON arm.
  - 8x8 grid, 1 resource, 3 hazards, 150 steps/ep, hazard_harm=0.02
  - resource_respawn_on_consume=True
  - 5 seeds (42, 43, 44, 45, 46)
  - Both HABIT and PLANNED mode for each seed (10 total runs)
  - Training: 300 eps per mode
  - Eval: 50 eps per mode (full agent pipeline)

Per seed, measure:
  PLANNED mode:
    - z_goal_norm (mean from training diagnostic)
    - z_harm_a_mean (mean z_harm_a norm during eval)
    - PLANNED_harm_rate, PLANNED_resource_rate
    - behavioral_equivalence_score = 1 - |HABIT_harm_rate - PLANNED_harm_rate|
  HABIT mode:
    - HABIT_harm_rate, HABIT_resource_rate

=== PRE-REGISTERED CRITERIA ===

PASS: >= 4/5 seeds show ALL of:
  C1: z_goal_norm < 0.1 (goal seeding collapsed)
  C2: |HABIT_harm_rate - PLANNED_harm_rate| < 0.02 (modes indistinguishable by harm_rate)

FAIL:
  < 4/5 seeds meeting both C1 and C2 -> does_not_support (attractor not robust)
  All C1 fail -> non_contributory (z_goal not deploying -- substrate limitation)
"""

import sys
import random
import json
import time
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


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_268_inv053_depression_attractor_replication"
CLAIM_IDS          = ["INV-053"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
GOAL_NORM_COLLAPSE_THRESH  = 0.1   # C1: z_goal_norm below this = seeding collapsed
HARM_RATE_EQUIV_THRESH     = 0.02  # C2: |PLANNED_hr - HABIT_hr| below this = modes indistinguishable
PASS_SEED_COUNT_THRESH     = 4     # need >= this many seeds meeting both C1 and C2

# ---------------------------------------------------------------------------
# Grid and episode parameters (LONG_HORIZON only)
# ---------------------------------------------------------------------------
GRID_SIZE       = 8
N_RESOURCES     = 1
N_HAZARDS       = 3
STEPS_PER_EP    = 150

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
TRAINING_EPISODES = 300
EVAL_EPISODES     = 50
SEEDS             = [42, 43, 44, 45, 46]
GREEDY_FRAC       = 0.4
MAX_BUF           = 4000
WF_BUF_MAX        = 2000
WORLD_DIM         = 32
BATCH_SIZE        = 16

# Learning rates
LR_E1      = 1e-3
LR_E2_WF   = 1e-3
LR_HARM    = 1e-4
LR_BENEFIT = 1e-3

# SD-018: resource proximity supervision
LAMBDA_RESOURCE = 0.5


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
    """Update z_goal from current step's benefit_exposure and drive_level."""
    b_exp = _get_benefit_exposure(obs_body)
    energy = _get_energy(obs_body)
    drive_level = max(0.0, 1.0 - energy)
    agent.update_z_goal(b_exp, drive_level=drive_level)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    """Create LONG_HORIZON environment matching EXQ-237a parameters."""
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=N_RESOURCES,
        num_hazards=N_HAZARDS,
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


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(condition: str, env: CausalGridWorldV2, seed: int) -> REEAgent:
    """
    HABIT mode:  z_goal disabled, hippocampal planning disabled.
                 Agent uses E1/E2 trajectories without goal-directed weighting.
    PLANNED mode: z_goal enabled, drive_weight=2.0, full hippocampal planning.
    """
    planned = (condition == "PLANNED")
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
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    condition: str,
    total_training_episodes: int,
    seed: int,
) -> Dict:
    """
    Mixed-policy training: 40% greedy, 60% random.

    Trains: E1 (prediction loss), E2 world_forward (explicit buffer),
    E3 harm_eval (stratified), E3 benefit_eval (proximity labels).
    PLANNED: additionally updates z_goal each step.
    """
    planned = (condition == "PLANNED")
    device  = agent.device
    n_act   = env.action_dim

    # Separate optimisers
    e1_params    = list(agent.e1.parameters())
    e2_wf_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    e1_opt      = optim.Adam(e1_params, lr=LR_E1)
    e2_wf_opt   = optim.Adam(e2_wf_params, lr=LR_E2_WF)
    harm_opt    = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_HARM)
    benefit_opt = optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=LR_BENEFIT)

    # Experience buffers
    wf_buf:       List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_pos_buf: List[torch.Tensor] = []
    harm_neg_buf: List[torch.Tensor] = []
    ben_zw_buf:   List[torch.Tensor] = []
    ben_lbl_buf:  List[float]        = []

    random.seed(seed)
    agent.train()

    for ep in range(total_training_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for step_i in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            # Sense
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            # E1 tick
            if ticks.get("e1_tick", False):
                _ = agent._e1_tick(latent)

            z_world_curr = latent.z_world.detach()

            # E2 world_forward buffer (from previous step)
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev, action_prev, z_world_curr))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            # Mixed training policy
            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_toward_resource(env)
            else:
                action_idx = random.randint(0, n_act - 1)
            action_oh = _onehot(action_idx, n_act, device)
            agent._last_action = action_oh

            # Benefit proximity label (before step)
            dist    = _dist_to_nearest_resource(env)
            is_near = 1.0 if dist <= 2 else 0.0

            # Env step
            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            # z_goal update (PLANNED only)
            if planned:
                _update_z_goal(agent, obs_dict["body_state"])

            # --- Train E1 ---
            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_opt.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
                    e1_opt.step()

            # --- SD-018: resource proximity supervision ---
            rfv = obs_dict.get("resource_field_view", None)
            if rfv is not None:
                rp_target = rfv[12].item()  # center cell = agent pos
                rp_loss = agent.compute_resource_proximity_loss(
                    rp_target, latent)
                if rp_loss.requires_grad:
                    e1_opt.zero_grad()
                    (LAMBDA_RESOURCE * rp_loss).backward()
                    torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
                    e1_opt.step()

            # --- Train E2 world_forward ---
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

            # --- Train E3 harm_eval (stratified) ---
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

            # --- Train E3 benefit_eval (proximity labels) ---
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

        if (ep + 1) % 50 == 0 or ep == total_training_episodes - 1:
            diag = agent.compute_goal_maintenance_diagnostic()
            print(
                f"  [train] seed={seed} ep {ep+1}/{total_training_episodes}"
                f" harm_pos={len(harm_pos_buf)}"
                f" harm_neg={len(harm_neg_buf)}"
                f" ben_samples={agent.e3._benefit_samples_seen}"
                f" goal_norm={diag['goal_norm']:.3f}",
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
    condition: str,
    eval_episodes: int,
    seed: int,
) -> Dict:
    """
    Eval using the full agent pipeline: generate_trajectories + select_action.

    Measures resource_rate, harm_rate, and z_harm_a_mean.
    """
    agent.eval()

    device    = agent.device
    world_dim = WORLD_DIM
    n_act     = env.action_dim

    resource_counts: List[int]   = []
    harm_rates:      List[float] = []
    z_harm_a_norms:  List[float] = []

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

            # E1 tick
            if ticks.get("e1_tick", False):
                e1_prior = agent._e1_tick(latent)
            else:
                e1_prior = torch.zeros(1, world_dim, device=device)

            # Full hippocampal planning
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks, temperature=1.0)

            if action is None:
                action = _onehot(random.randint(0, n_act - 1), n_act, device)
                agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)

            # Record z_harm_a norm if available
            if latent.z_harm_a is not None:
                z_harm_a_norms.append(float(latent.z_harm_a.norm().item()))

            if info.get("transition_type") == "resource":
                ep_resources += 1
            if float(harm_signal) < 0:
                ep_harm_sum += abs(float(harm_signal))
            ep_steps += 1

            # z_goal maintenance during eval (PLANNED only, no gradients)
            if condition == "PLANNED":
                with torch.no_grad():
                    _update_z_goal(agent, obs_dict["body_state"])

            if done:
                break

        resource_counts.append(1 if ep_resources >= 1 else 0)
        harm_rates.append(ep_harm_sum / max(1, ep_steps))

    resource_rate = sum(resource_counts) / max(1, len(resource_counts))
    harm_rate     = sum(harm_rates)      / max(1, len(harm_rates))
    z_harm_a_mean = (
        sum(z_harm_a_norms) / max(1, len(z_harm_a_norms))
        if z_harm_a_norms else 0.0
    )

    return {
        "resource_rate": resource_rate,
        "harm_rate":     harm_rate,
        "z_harm_a_mean": z_harm_a_mean,
    }


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def _run_seed(
    seed:                    int,
    total_training_episodes: int,
    eval_episodes:           int,
) -> Dict:
    """Run HABIT and PLANNED conditions for one seed in LONG_HORIZON context."""
    results: Dict = {}

    for condition in ["HABIT", "PLANNED"]:
        print(
            f"\n[V3-EXQ-268] Seed {seed} Mode {condition}"
            f" steps={STEPS_PER_EP}",
            flush=True,
        )
        env   = _make_env(seed)
        agent = _make_agent(condition, env, seed)

        train_res = _train(agent, env, condition, total_training_episodes, seed)
        eval_res  = _eval(agent, env, condition, eval_episodes, seed)

        print(
            f"  [eval done] seed={seed} mode={condition}"
            f" resource_rate={eval_res['resource_rate']:.3f}"
            f" harm_rate={eval_res['harm_rate']:.5f}"
            f" z_harm_a_mean={eval_res['z_harm_a_mean']:.4f}"
            f" goal_norm={train_res['goal_norm']:.3f}",
            flush=True,
        )

        results[condition] = {
            "resource_rate": eval_res["resource_rate"],
            "harm_rate":     eval_res["harm_rate"],
            "z_harm_a_mean": eval_res["z_harm_a_mean"],
            "goal_norm":     train_res["goal_norm"],
        }

    return results


# ---------------------------------------------------------------------------
# Aggregate and criteria
# ---------------------------------------------------------------------------

def _aggregate(all_results: List[Dict], seeds: List[int]) -> Dict:
    """
    Compute per-seed and aggregate metrics.

    PASS criterion: >= PASS_SEED_COUNT_THRESH seeds meeting ALL of:
      C1: z_goal_norm < GOAL_NORM_COLLAPSE_THRESH
      C2: |PLANNED_harm_rate - HABIT_harm_rate| < HARM_RATE_EQUIV_THRESH
    """
    per_seed = []

    for i, r in enumerate(all_results):
        habit = r["HABIT"]
        plan  = r["PLANNED"]

        harm_rate_diff = abs(plan["harm_rate"] - habit["harm_rate"])
        behav_equiv_score = 1.0 - harm_rate_diff  # higher = more equivalent

        c1_met   = plan["goal_norm"] < GOAL_NORM_COLLAPSE_THRESH
        c2_met   = harm_rate_diff    < HARM_RATE_EQUIV_THRESH
        both_met = c1_met and c2_met

        per_seed.append({
            "seed":                         seeds[i],
            "z_goal_norm":                  plan["goal_norm"],
            "z_harm_a_mean":                plan["z_harm_a_mean"],
            "PLANNED_harm_rate":            plan["harm_rate"],
            "PLANNED_resource_rate":        plan["resource_rate"],
            "HABIT_harm_rate":              habit["harm_rate"],
            "HABIT_resource_rate":          habit["resource_rate"],
            "harm_rate_diff":               harm_rate_diff,
            "behavioral_equivalence_score": behav_equiv_score,
            "c1_goal_norm_met":             c1_met,
            "c2_harm_rate_equiv_met":       c2_met,
            "both_criteria_met":            both_met,
        })

    def _mean(key):
        return sum(s[key] for s in per_seed) / max(1, len(per_seed))

    pass_count = sum(1 for s in per_seed if s["both_criteria_met"])
    c1_count   = sum(1 for s in per_seed if s["c1_goal_norm_met"])
    c2_count   = sum(1 for s in per_seed if s["c2_harm_rate_equiv_met"])

    overall_pass = pass_count >= PASS_SEED_COUNT_THRESH

    return {
        "per_seed":    per_seed,
        "n_seeds":     len(per_seed),
        "means": {
            "z_goal_norm":                  _mean("z_goal_norm"),
            "z_harm_a_mean":                _mean("z_harm_a_mean"),
            "PLANNED_harm_rate":            _mean("PLANNED_harm_rate"),
            "PLANNED_resource_rate":        _mean("PLANNED_resource_rate"),
            "HABIT_harm_rate":              _mean("HABIT_harm_rate"),
            "HABIT_resource_rate":          _mean("HABIT_resource_rate"),
            "harm_rate_diff":               _mean("harm_rate_diff"),
            "behavioral_equivalence_score": _mean("behavioral_equivalence_score"),
        },
        "pass_count_by_criterion": {
            "c1_goal_norm_below_thresh":      c1_count,
            "c2_harm_rate_equiv_below_thresh": c2_count,
            "both_criteria":                  pass_count,
        },
        "overall_pass": overall_pass,
        "pass_count":   pass_count,
        "c1_count":     c1_count,
        "c2_count":     c2_count,
    }


def _decide(agg: Dict) -> Tuple[str, str, str]:
    """Return (outcome, evidence_direction, decision)."""
    c1_count = agg["c1_count"]

    if c1_count == 0:
        return "FAIL", "non_contributory", "substrate_limitation"
    if agg["overall_pass"]:
        return "PASS", "supports", "retain_ree"
    return "FAIL", "does_not_support", "attractor_not_robust"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    training = 5     if args.dry_run else TRAINING_EPISODES
    n_eval   = 5     if args.dry_run else EVAL_EPISODES
    seeds    = [42]  if args.dry_run else SEEDS

    print(
        f"[V3-EXQ-268] INV-053: Depression Attractor Replication (5-seed)"
        f" dry_run={args.dry_run}"
        f" training={training} eval={n_eval} seeds={seeds}",
        flush=True,
    )
    print(
        f"  LONG_HORIZON: {N_RESOURCES} resource, {N_HAZARDS} hazards,"
        f" {STEPS_PER_EP} steps, hazard_harm=0.02",
        flush=True,
    )
    print(
        f"  PASS criterion: >={PASS_SEED_COUNT_THRESH}/{len(seeds)} seeds with"
        f" z_goal_norm<{GOAL_NORM_COLLAPSE_THRESH}"
        f" AND |PLANNED_hr-HABIT_hr|<{HARM_RATE_EQUIV_THRESH}",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        print(f"\nSeed {seed}", flush=True)
        seed_results = _run_seed(
            seed=seed,
            total_training_episodes=training,
            eval_episodes=n_eval,
        )
        all_results.append(seed_results)

        # Per-seed interim verdict
        plan  = seed_results["PLANNED"]
        habit = seed_results["HABIT"]
        hr_diff = abs(plan["harm_rate"] - habit["harm_rate"])
        c1 = plan["goal_norm"] < GOAL_NORM_COLLAPSE_THRESH
        c2 = hr_diff < HARM_RATE_EQUIV_THRESH
        seed_passed = c1 and c2
        print(
            f"verdict: {'PASS' if seed_passed else 'FAIL'}"
            f" seed={seed} c1={c1} c2={c2}"
            f" z_goal_norm={plan['goal_norm']:.3f}"
            f" hr_diff={hr_diff:.5f}",
            flush=True,
        )

    # Aggregate
    agg = _aggregate(all_results, seeds)
    m   = agg["means"]
    pcc = agg["pass_count_by_criterion"]

    outcome, direction, decision = _decide(agg)
    experiment_passes = agg["overall_pass"]
    seeds_passing     = agg["pass_count"]

    print(f"\n[V3-EXQ-268] === Results ===", flush=True)
    print(
        f"  Mean z_goal_norm={m['z_goal_norm']:.3f}"
        f"  (C1 thresh < {GOAL_NORM_COLLAPSE_THRESH})",
        flush=True,
    )
    print(
        f"  Mean HABIT_hr={m['HABIT_harm_rate']:.5f}"
        f"  PLANNED_hr={m['PLANNED_harm_rate']:.5f}"
        f"  hr_diff={m['harm_rate_diff']:.5f}"
        f"  (C2 thresh < {HARM_RATE_EQUIV_THRESH})",
        flush=True,
    )
    print(
        f"  Mean PLANNED_rr={m['PLANNED_resource_rate']:.3f}"
        f"  HABIT_rr={m['HABIT_resource_rate']:.3f}"
        f"  behav_equiv_score={m['behavioral_equivalence_score']:.4f}",
        flush=True,
    )
    print(
        f"  Mean z_harm_a_mean={m['z_harm_a_mean']:.4f}",
        flush=True,
    )
    print(
        f"  C1 (z_goal_norm < {GOAL_NORM_COLLAPSE_THRESH}): {pcc['c1_goal_norm_below_thresh']}/{agg['n_seeds']} seeds",
        flush=True,
    )
    print(
        f"  C2 (|PLANNED_hr-HABIT_hr| < {HARM_RATE_EQUIV_THRESH}): {pcc['c2_harm_rate_equiv_below_thresh']}/{agg['n_seeds']} seeds",
        flush=True,
    )
    print(
        f"  Both criteria: {pcc['both_criteria']}/{agg['n_seeds']} seeds"
        f"  (need >={PASS_SEED_COUNT_THRESH})",
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
    ts      = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
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
        "timestamp_utc":      ts,
        "seeds":              seeds,
        # Parameters
        "training_episodes":  training,
        "eval_episodes":      n_eval,
        "steps_per_ep":       STEPS_PER_EP,
        "n_resources":        N_RESOURCES,
        "n_hazards":          N_HAZARDS,
        "hazard_harm":        0.02,
        "greedy_frac":        GREEDY_FRAC,
        "world_dim":          WORLD_DIM,
        "drive_weight":       2.0,
        "use_resource_proximity_head": True,
        # Thresholds
        "goal_norm_collapse_thresh":  GOAL_NORM_COLLAPSE_THRESH,
        "harm_rate_equiv_thresh":     HARM_RATE_EQUIV_THRESH,
        "pass_seed_count_thresh":     PASS_SEED_COUNT_THRESH,
        # Summary
        "seeds_passing":      seeds_passing,
        "experiment_passes":  experiment_passes,
        # Aggregate means
        "mean_z_goal_norm":                  float(m["z_goal_norm"]),
        "mean_z_harm_a_mean":                float(m["z_harm_a_mean"]),
        "mean_PLANNED_harm_rate":            float(m["PLANNED_harm_rate"]),
        "mean_PLANNED_resource_rate":        float(m["PLANNED_resource_rate"]),
        "mean_HABIT_harm_rate":              float(m["HABIT_harm_rate"]),
        "mean_HABIT_resource_rate":          float(m["HABIT_resource_rate"]),
        "mean_harm_rate_diff":               float(m["harm_rate_diff"]),
        "mean_behavioral_equivalence_score": float(m["behavioral_equivalence_score"]),
        # Criteria counts
        "pass_count_by_criterion": pcc,
        "overall_pass":            agg["overall_pass"],
        # Per-seed detail
        "per_seed_results": [
            {
                "seed":                         s["seed"],
                "z_harm_a_mean":                s["z_harm_a_mean"],
                "z_goal_norm":                  s["z_goal_norm"],
                "HABIT_harm_rate":              s["HABIT_harm_rate"],
                "PLANNED_harm_rate":            s["PLANNED_harm_rate"],
                "behavioral_equivalence_score": s["behavioral_equivalence_score"],
                "harm_rate_diff":               s["harm_rate_diff"],
                "PLANNED_resource_rate":        s["PLANNED_resource_rate"],
                "HABIT_resource_rate":          s["HABIT_resource_rate"],
                "c1_goal_norm_met":             s["c1_goal_norm_met"],
                "c2_harm_rate_equiv_met":       s["c2_harm_rate_equiv_met"],
                "both_criteria_met":            s["both_criteria_met"],
            }
            for s in agg["per_seed"]
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-268] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
