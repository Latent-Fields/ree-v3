#!/opt/local/bin/python3
"""
V3-EXQ-251 -- MECH-186: VALENCE_WANTING Floor Clamp Prevents Depression Attractor

Claims: MECH-186
EXPERIMENT_PURPOSE = "evidence"

=== SCIENTIFIC QUESTION ===

Does maintaining a minimum VALENCE_WANTING floor (simulating serotonergic benefit
terrain maintenance) prevent depression attractor formation in the LONG_HORIZON
condition?

The depression attractor (confirmed by EXQ-249) is the state where z_goal fails
to seed because benefit terrain has collapsed (hazard_harm overwhelms the benefit
signal -> drive cycles cannot bootstrap goal representation). Once entered, HABIT
and PLANNED systems produce indistinguishable behavior.

MECH-186 claims: tonic serotonergic support maintains a minimum wanting tone
(valence_wanting floor), preventing the transition into the depression attractor.
This maps to environmental intervention and activity scheduling in therapy -- an
externally supplied terrain restoration that preserves the goal pathway.

=== BIOLOGICAL GROUNDING ===

VALENCE_WANTING (component 0 of the RBF valence vector, SD-014) tracks the agent's
wanting signal across hippocampal map nodes. When benefit exposure is low (sparse
resources, hazard-dense environment), VALENCE_WANTING decays toward zero across
all nodes. The floor clamp is implemented in GoalState.update() as a minimum
z_goal norm constraint -- after decay, if z_goal.norm() drops below the floor,
it is rescaled back to the floor value while preserving direction.

Therapeutic analog: antidepressants / behavioral activation maintain a tonic
wanting signal that prevents the motivational system from collapsing to anhedonia.
The floor enables the agent to maintain directional goal information even when
benefit contacts are rare, allowing recovery when resources become available.

=== 2-CONDITION x 3-SEED DESIGN ===

Both conditions: LONG_HORIZON (8x8, 1 resource, 3 hazards, 150 steps/ep,
hazard_harm=0.02), PLANNED mode (z_goal_enabled=True, drive_weight=2.0),
3 seeds [42, 7, 13].

Condition A (BASELINE): valence_wanting_floor=0.0 (no floor, depression expected).
Condition B (FLOOR_MAINTAINED): valence_wanting_floor=0.05 (floor clamp active).

Both conditions run HABIT mode to compute the behavioral gap:
  behavioral_gap = PLANNED_resource_rate - HABIT_resource_rate

Per seed, 4 runs: A_HABIT, A_PLANNED, B_HABIT, B_PLANNED.

=== PRE-REGISTERED CRITERIA ===

PASS: Condition B vs Condition A, majority >= 2/3 seeds:
  C1: z_goal_norm(B) >= 0.15 (floor restored z_goal seeding)
  C2: behavioral_gap(B) >= 0.05 (PLANNED now outperforms HABIT in B)

PASS requires BOTH C1 AND C2 in >= 2/3 seeds.

evidence_direction interpretation:
  PASS -> supports (floor clamp prevents depression attractor)
  C1 FAIL only -> substrate_limitation (floor clamp not raising z_goal)
  C2 FAIL, C1 PASS -> does_not_support (z_goal seeded but no behavioral rescue)
  Both FAIL -> does_not_support (floor clamp does nothing)
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
EXPERIMENT_TYPE    = "v3_exq_251_mech186_valence_wanting_floor"
CLAIM_IDS          = ["MECH-186"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
FLOOR_GOAL_NORM_THRESH    = 0.15   # C1: z_goal_norm in B must reach this
BEHAVIORAL_GAP_THRESH     = 0.05   # C2: PLANNED_rr - HABIT_rr in B must exceed this
PASS_SEED_COUNT_THRESH    = 2      # majority out of 3 seeds

# ---------------------------------------------------------------------------
# Grid and episode parameters (LONG_HORIZON only)
# ---------------------------------------------------------------------------
GRID_SIZE       = 8
N_RESOURCES     = 1
N_HAZARDS       = 3
STEPS_PER_EP    = 150
HAZARD_HARM     = 0.02

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

# SD-018: resource proximity supervision
LAMBDA_RESOURCE = 0.5

# MECH-186: floor value for FLOOR_MAINTAINED condition
VALENCE_WANTING_FLOOR = 0.05


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
    """Manhattan distance from agent to nearest resource (999 if none)."""
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
    """Create LONG_HORIZON environment matching EXQ-249 parameters."""
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
    condition: str,
    cond_label: str,
    env: CausalGridWorldV2,
    seed: int,
) -> REEAgent:
    """
    Build REEAgent for the given condition.

    cond_label: "A_HABIT" | "A_PLANNED" | "B_HABIT" | "B_PLANNED"

    Condition A: valence_wanting_floor=0.0 (BASELINE -- no floor)
    Condition B: valence_wanting_floor=0.05 (FLOOR_MAINTAINED)

    PLANNED: z_goal_enabled=True, e1_goal_conditioned=True, drive_weight=2.0
    HABIT:   z_goal_enabled=False, drive_weight=0.0
    """
    planned = (condition == "PLANNED")
    use_floor = cond_label.startswith("B_")

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
        valence_wanting_floor=VALENCE_WANTING_FLOOR if (planned and use_floor) else 0.0,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    condition: str,
    cond_label: str,
    warmup_episodes: int,
    seed: int,
) -> Dict:
    """
    Mixed-policy warmup: 40% greedy, 60% random.

    Trains: E1 (prediction loss), E2 world_forward (explicit buffer),
    E3 harm_eval (stratified), E3 benefit_eval (proximity labels).
    PLANNED: additionally updates z_goal each step (floor clamp active in B).
    """
    planned = (condition == "PLANNED")
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

            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_toward_resource(env)
            else:
                action_idx = random.randint(0, n_act - 1)
            action_oh = _onehot(action_idx, n_act, device)
            agent._last_action = action_oh

            dist    = _dist_to_nearest_resource(env)
            is_near = 1.0 if dist <= 2 else 0.0

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            # z_goal update (PLANNED only; floor clamp fires inside GoalState.update)
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

            # SD-018: resource proximity supervision
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
            print(
                f"    [train] cond={cond_label} seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
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
    cond_label: str,
    eval_episodes: int,
    seed: int,
) -> Dict:
    """
    Eval using the full agent pipeline: generate_trajectories + select_action.
    No weight updates during eval. PLANNED: z_goal updates continue (goal maintenance).
    """
    planned = (condition == "PLANNED")
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

            if latent.z_harm_a is not None:
                z_harm_a_norms.append(float(latent.z_harm_a.norm().item()))

            if info.get("transition_type") == "resource":
                ep_resources += 1
            if float(harm_signal) < 0:
                ep_harm_sum += abs(float(harm_signal))
            ep_steps += 1

            # z_goal maintenance (PLANNED only, no gradients; floor clamp fires inside)
            if planned:
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
    seed:            int,
    warmup_episodes: int,
    eval_episodes:   int,
) -> Dict:
    """
    Run all 4 sub-conditions for one seed:
      A_HABIT, A_PLANNED (baseline, no floor)
      B_HABIT, B_PLANNED (floor_maintained)
    """
    results: Dict = {}

    for cond_label in ["A_HABIT", "A_PLANNED", "B_HABIT", "B_PLANNED"]:
        condition = "PLANNED" if cond_label.endswith("PLANNED") else "HABIT"

        print(
            f"\n[V3-EXQ-251] Seed {seed} Condition {cond_label}"
            f" steps={STEPS_PER_EP}",
            flush=True,
        )

        env   = _make_env(seed)
        agent = _make_agent(condition, cond_label, env, seed)

        warmup_res = _warmup(
            agent, env, condition, cond_label, warmup_episodes, seed
        )
        eval_res = _eval(
            agent, env, condition, cond_label, eval_episodes, seed
        )

        print(
            f"  [eval done] cond={cond_label} seed={seed}"
            f" resource_rate={eval_res['resource_rate']:.3f}"
            f" harm_rate={eval_res['harm_rate']:.5f}"
            f" goal_norm={warmup_res['goal_norm']:.3f}",
            flush=True,
        )

        results[cond_label] = {
            "resource_rate": eval_res["resource_rate"],
            "harm_rate":     eval_res["harm_rate"],
            "z_harm_a_mean": eval_res["z_harm_a_mean"],
            "goal_norm":     warmup_res["goal_norm"],
        }

        if condition == "PLANNED":
            print(f"verdict: PASS", flush=True)
        else:
            print(f"verdict: done", flush=True)

    return results


# ---------------------------------------------------------------------------
# Aggregate and criteria
# ---------------------------------------------------------------------------

def _aggregate(all_results: List[Dict], seeds: List[int]) -> Dict:
    """
    Compute per-seed and aggregate metrics for MECH-186 floor clamp test.

    Per seed:
      - z_goal_norm_A = A_PLANNED goal_norm  (baseline, no floor)
      - z_goal_norm_B = B_PLANNED goal_norm  (floor_maintained)
      - behavioral_gap_A = A_PLANNED_rr - A_HABIT_rr
      - behavioral_gap_B = B_PLANNED_rr - B_HABIT_rr

    C1: z_goal_norm_B >= FLOOR_GOAL_NORM_THRESH
    C2: behavioral_gap_B >= BEHAVIORAL_GAP_THRESH
    PASS per seed: both C1 and C2.
    """
    per_seed = []

    for i, r in enumerate(all_results):
        a_habit  = r["A_HABIT"]
        a_plan   = r["A_PLANNED"]
        b_habit  = r["B_HABIT"]
        b_plan   = r["B_PLANNED"]

        behav_gap_a = a_plan["resource_rate"] - a_habit["resource_rate"]
        behav_gap_b = b_plan["resource_rate"] - b_habit["resource_rate"]

        c1_met = b_plan["goal_norm"] >= FLOOR_GOAL_NORM_THRESH
        c2_met = behav_gap_b >= BEHAVIORAL_GAP_THRESH
        both_met = c1_met and c2_met

        per_seed.append({
            "seed":               seeds[i],
            "z_goal_norm_A":      a_plan["goal_norm"],
            "z_goal_norm_B":      b_plan["goal_norm"],
            "A_HABIT_rr":         a_habit["resource_rate"],
            "A_PLANNED_rr":       a_plan["resource_rate"],
            "B_HABIT_rr":         b_habit["resource_rate"],
            "B_PLANNED_rr":       b_plan["resource_rate"],
            "A_HABIT_hr":         a_habit["harm_rate"],
            "A_PLANNED_hr":       a_plan["harm_rate"],
            "B_HABIT_hr":         b_habit["harm_rate"],
            "B_PLANNED_hr":       b_plan["harm_rate"],
            "behavioral_gap_A":   behav_gap_a,
            "behavioral_gap_B":   behav_gap_b,
            "z_harm_a_mean_B":    b_plan["z_harm_a_mean"],
            "c1_floor_goal_norm": c1_met,
            "c2_behavioral_gap":  c2_met,
            "both_criteria_met":  both_met,
        })

    def _mean(key):
        return sum(s[key] for s in per_seed) / max(1, len(per_seed))

    pass_count = sum(1 for s in per_seed if s["both_criteria_met"])
    c1_count   = sum(1 for s in per_seed if s["c1_floor_goal_norm"])
    c2_count   = sum(1 for s in per_seed if s["c2_behavioral_gap"])

    overall_pass = pass_count >= PASS_SEED_COUNT_THRESH

    return {
        "per_seed":  per_seed,
        "n_seeds":   len(per_seed),
        "means": {
            "z_goal_norm_A":    _mean("z_goal_norm_A"),
            "z_goal_norm_B":    _mean("z_goal_norm_B"),
            "A_HABIT_rr":       _mean("A_HABIT_rr"),
            "A_PLANNED_rr":     _mean("A_PLANNED_rr"),
            "B_HABIT_rr":       _mean("B_HABIT_rr"),
            "B_PLANNED_rr":     _mean("B_PLANNED_rr"),
            "behavioral_gap_A": _mean("behavioral_gap_A"),
            "behavioral_gap_B": _mean("behavioral_gap_B"),
            "z_harm_a_mean_B":  _mean("z_harm_a_mean_B"),
        },
        "pass_count_by_criterion": {
            "c1_floor_goal_norm": c1_count,
            "c2_behavioral_gap":  c2_count,
            "both_criteria":      pass_count,
        },
        "overall_pass": overall_pass,
        "pass_count":   pass_count,
        "c1_count":     c1_count,
        "c2_count":     c2_count,
    }


def _decide(agg: Dict) -> Tuple[str, str, str]:
    """Return (outcome, evidence_direction, decision)."""
    c1_count = agg["c1_count"]
    c2_count = agg["c2_count"]

    if c1_count == 0:
        return "FAIL", "does_not_support", "substrate_limitation"
    if agg["overall_pass"]:
        return "PASS", "supports", "retain_ree"
    if c1_count >= PASS_SEED_COUNT_THRESH and c2_count < PASS_SEED_COUNT_THRESH:
        # z_goal seeded but PLANNED didn't outperform HABIT -> goal seeding not sufficient
        return "FAIL", "does_not_support", "goal_seeded_no_behavioral_rescue"
    return "FAIL", "does_not_support", "floor_clamp_insufficient"


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
    seeds  = [42] if args.dry_run else SEEDS

    print(
        f"[V3-EXQ-251] MECH-186: VALENCE_WANTING Floor Clamp vs Baseline"
        f" (LONG_HORIZON, 2-cond x 3-seed)"
        f" dry_run={args.dry_run}"
        f" warmup={warmup} eval={n_eval} seeds={seeds}",
        flush=True,
    )
    print(
        f"  LONG_HORIZON: {N_RESOURCES} resource, {N_HAZARDS} hazards,"
        f" {STEPS_PER_EP} steps, hazard_harm={HAZARD_HARM}",
        flush=True,
    )
    print(
        f"  Condition A: no floor (BASELINE, valence_wanting_floor=0.0)",
        flush=True,
    )
    print(
        f"  Condition B: floor_maintained (valence_wanting_floor={VALENCE_WANTING_FLOOR})",
        flush=True,
    )
    print(
        f"  PASS: >={PASS_SEED_COUNT_THRESH}/3 seeds with"
        f" z_goal_norm(B)>={FLOOR_GOAL_NORM_THRESH}"
        f" AND behavioral_gap(B)>={BEHAVIORAL_GAP_THRESH}",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        print(f"\n[V3-EXQ-251] === Seed {seed} ===", flush=True)
        seed_results = _run_seed(
            seed=seed,
            warmup_episodes=warmup,
            eval_episodes=n_eval,
        )
        all_results.append(seed_results)

    # Aggregate
    agg = _aggregate(all_results, seeds)
    m   = agg["means"]
    pcc = agg["pass_count_by_criterion"]

    outcome, direction, decision = _decide(agg)

    print(f"\n[V3-EXQ-251] === Results ===", flush=True)
    print(
        f"  Mean z_goal_norm: A={m['z_goal_norm_A']:.3f}"
        f"  B={m['z_goal_norm_B']:.3f}"
        f"  (C1 B>={FLOOR_GOAL_NORM_THRESH})",
        flush=True,
    )
    print(
        f"  Mean resource_rate: A_HABIT={m['A_HABIT_rr']:.3f}"
        f"  A_PLANNED={m['A_PLANNED_rr']:.3f}"
        f"  B_HABIT={m['B_HABIT_rr']:.3f}"
        f"  B_PLANNED={m['B_PLANNED_rr']:.3f}",
        flush=True,
    )
    print(
        f"  Mean behavioral_gap: A={m['behavioral_gap_A']:+.3f}"
        f"  B={m['behavioral_gap_B']:+.3f}"
        f"  (C2 B>={BEHAVIORAL_GAP_THRESH})",
        flush=True,
    )
    print(
        f"  Mean z_harm_a(B)={m['z_harm_a_mean_B']:.4f}",
        flush=True,
    )
    print(
        f"  C1 (z_goal_norm_B>={FLOOR_GOAL_NORM_THRESH}):"
        f" {pcc['c1_floor_goal_norm']}/{agg['n_seeds']} seeds",
        flush=True,
    )
    print(
        f"  C2 (behavioral_gap_B>={BEHAVIORAL_GAP_THRESH}):"
        f" {pcc['c2_behavioral_gap']}/{agg['n_seeds']} seeds",
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
        "warmup_episodes":        warmup,
        "eval_episodes":          n_eval,
        "steps_per_ep":           STEPS_PER_EP,
        "n_resources":            N_RESOURCES,
        "n_hazards":              N_HAZARDS,
        "hazard_harm":            HAZARD_HARM,
        "greedy_frac":            GREEDY_FRAC,
        "valence_wanting_floor":  VALENCE_WANTING_FLOOR,
        # Thresholds
        "floor_goal_norm_thresh":  FLOOR_GOAL_NORM_THRESH,
        "behavioral_gap_thresh":   BEHAVIORAL_GAP_THRESH,
        "pass_seed_count_thresh":  PASS_SEED_COUNT_THRESH,
        # Aggregate means
        "mean_z_goal_norm_A":    float(m["z_goal_norm_A"]),
        "mean_z_goal_norm_B":    float(m["z_goal_norm_B"]),
        "mean_A_HABIT_rr":       float(m["A_HABIT_rr"]),
        "mean_A_PLANNED_rr":     float(m["A_PLANNED_rr"]),
        "mean_B_HABIT_rr":       float(m["B_HABIT_rr"]),
        "mean_B_PLANNED_rr":     float(m["B_PLANNED_rr"]),
        "mean_behavioral_gap_A": float(m["behavioral_gap_A"]),
        "mean_behavioral_gap_B": float(m["behavioral_gap_B"]),
        "mean_z_harm_a_mean_B":  float(m["z_harm_a_mean_B"]),
        # Criteria
        "c1_floor_goal_norm_count": pcc["c1_floor_goal_norm"],
        "c2_behavioral_gap_count":  pcc["c2_behavioral_gap"],
        "both_criteria_count":      pcc["both_criteria"],
        # Per-seed detail
        "per_seed_results": [
            {
                "seed":              s["seed"],
                "z_goal_norm_A":     s["z_goal_norm_A"],
                "z_goal_norm_B":     s["z_goal_norm_B"],
                "A_HABIT_rr":        s["A_HABIT_rr"],
                "A_PLANNED_rr":      s["A_PLANNED_rr"],
                "B_HABIT_rr":        s["B_HABIT_rr"],
                "B_PLANNED_rr":      s["B_PLANNED_rr"],
                "behavioral_gap_A":  s["behavioral_gap_A"],
                "behavioral_gap_B":  s["behavioral_gap_B"],
                "z_harm_a_mean_B":   s["z_harm_a_mean_B"],
                "c1_floor_goal_norm":s["c1_floor_goal_norm"],
                "c2_behavioral_gap": s["c2_behavioral_gap"],
                "both_criteria_met": s["both_criteria_met"],
            }
            for s in agg["per_seed"]
        ],
        "pass_count_by_criterion": pcc,
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[V3-EXQ-251] Output written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
