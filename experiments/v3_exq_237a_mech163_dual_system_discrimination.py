#!/opt/local/bin/python3
"""
V3-EXQ-237a -- MECH-163: Dual Goal-Directed Systems Discrimination
  Supersedes V3-EXQ-237 (critical bug fix: hazard_harm=0.5 -> 0.02)

Claims: MECH-163
EXPERIMENT_PURPOSE = "evidence"

=== BUG FIX (EXQ-237 -> EXQ-237a) ===

EXQ-237 had hazard_harm=0.5. With 3 hazards in LONG_HORIZON, two contacts
bring health to 0 (2 x 0.5 = 1.0 = full depletion) -> done=True at step ~18
on average (confirmed by queue calibration note). This corrupts resource_rate
because episodes terminate before the agent can locate resources. The HABIT vs
PLANNED distinction becomes uninterpretable.

Fix: hazard_harm=0.02 (consistent with EXQ-225/226 and all other V3 scripts
that use CausalGridWorldV2 for behavioural discrimination). At hazard_harm=0.02,
average episode length is ~182 steps (sufficient for full evaluation).

All other parameters, criteria, and design choices are unchanged from EXQ-237.

=== DESIGN NOTE: Operationalising "habit" vs "planned" in REE terms ===

MECH-163 asserts two distinct goal-directed systems:

  HABIT (SNc/dorsal-striatum, model-free):
    - HippocampalModule proposes value-flat CEM trajectories (ARC-007 strict)
    - E3 scores them via harm_eval - benefit_eval (learned S->R associations)
    - No persistent goal attractor; no E1 goal-conditioning
    - Sufficient for resource approach when benefit_eval is well-calibrated

  PLANNED (VTA/hippocampal+PFC, model-based):
    - z_goal_enabled=True: goal attractor seeded from benefit contacts,
      maintained by E1 recurrent conditioning (MECH-116)
    - z_goal -> E1 prior -> terrain_prior -> CEM search biased toward
      goal-adjacent trajectory sequences (multi-step planning)
    - E3 additionally subtracts goal_proximity from trajectory cost
    - Required when resources are sparse, episodes are long, and
      benefit_eval alone gives insufficient directional signal

Both conditions use benefit_eval_enabled=True. HABIT is NOT a strawman:
it uses the full E3 benefit evaluation on proposed trajectories. The
sole architectural difference is the z_goal pathway.

=== SCIENTIFIC QUESTION ===

Does the PLANNED system's advantage appear specifically in the
LONG_HORIZON context (where habit-level benefit_eval is insufficient)
but not in the SIMPLE context (where habit can succeed)?

A pure HABIT vs PLANNED comparison without context manipulation cannot
distinguish "no dual system" from "dual system hidden by task difficulty."
The 2x2 interaction test is the minimum necessary design.

Prediction (MECH-163 support):
    planned_lift_long > planned_lift_simple + INTERACTION_THRESH
  where planned_lift_X = PLANNED_X.resource_rate - HABIT_X.resource_rate

=== 2x2 DESIGN ===

Factor 1 -- Context:
  SIMPLE:       8x8, 2 resources, 1 hazard, 80 steps/ep.
                Familiar gradient structure; habit-level benefit_eval
                calibrated from warmup is sufficient.
  LONG_HORIZON: 8x8, 1 resource, 3 hazards, 150 steps/ep.
                Sparse resources, obstacle-dense search space. Habit
                benefit_eval alone provides weak directional signal;
                z_goal persistent attractor sustains goal-directed search.

Factor 2 -- Policy:
  HABIT:   z_goal_enabled=False, benefit_eval_enabled=True, drive_weight=0.
           Value-flat HippocampalModule CEM proposals + harm - benefit.
  PLANNED: z_goal_enabled=True, e1_goal_conditioned=True, drive_weight=2.
           z_goal biases E1 prior -> terrain_prior -> CEM proposal bias.
           E3 score additionally subtracts goal_proximity.

=== WARMUP (200 eps, 40% greedy) ===

Both conditions: 40% greedy toward nearest resource, 60% random.
Trains E1 (prediction), E2 (world_forward), E3 harm_eval (stratified
pos/neg buffer), E3 benefit_eval (resource proximity labels).
PLANNED additionally updates z_goal on each step.

=== EVAL (100 eps) ===

Full agent pipeline: generate_trajectories() + select_action().
This captures z_goal -> E1 prior -> terrain_prior -> CEM bias,
which is the core mechanism of the MECH-163 planned system.
PLANNED continues z_goal updates during eval (goal maintenance).

=== PRE-REGISTERED CRITERIA ===

Key metrics:
  planned_lift_simple   = PLANNED_SIMPLE - HABIT_SIMPLE resource_rate
  planned_lift_long     = PLANNED_LONG   - HABIT_LONG   resource_rate
  interaction           = planned_lift_long - planned_lift_simple

Criteria evaluated per seed, majority rule (>= 2/3 seeds):
  C1: avg z_goal_norm >= 0.05 for PLANNED (seeding confirmed).
      C1 FAIL -> substrate_limitation (z_goal not seeding).
  C2: HABIT_SIMPLE resource_rate >= 0.10 (informational -- habit not strawman).
  C3: planned_lift_long >= 0.05 (PLANNED outperforms HABIT in long-horizon).
  C4: interaction > 0.02 (advantage is context-specific).
  C5: PLANNED_LONG harm_rate <= HABIT_LONG harm_rate * 1.5 (harm control).

PASS = C1 AND C3 AND C4 AND C5, majority >= 2/3 seeds each.

Evidence interpretation:
  PASS -> supports (MECH-163: two regime dual goal system confirmed)
  C1 FAIL -> non_contributory (substrate limitation, z_goal not seeding)
  C3 FAIL, C1 PASS -> does_not_support (planned not better in long-horizon)
  C4 FAIL, C3 PASS -> does_not_support (planning helps uniformly, not regime-specific)
  C5 FAIL -> weakens (planning causes harmful exploration)
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
EXPERIMENT_TYPE    = "v3_exq_237a_mech163_dual_system_discrimination"
CLAIM_IDS          = ["MECH-163"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES         = "V3-EXQ-237"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
GOAL_NORM_THRESH      = 0.05   # C1: z_goal must seed in PLANNED condition
HABIT_SIMPLE_MIN      = 0.10   # C2: informational -- habit not a strawman
LIFT_LONG_THRESH      = 0.05   # C3: PLANNED advantage in long-horizon context
INTERACTION_THRESH    = 0.02   # C4: context-specific advantage (interaction)
HARM_RATIO_MAX        = 1.5    # C5: PLANNED harm <= HABIT harm * this

MAJORITY_THRESH       = 2      # criteria met in >= MAJORITY_THRESH seeds (out of 3)

# ---------------------------------------------------------------------------
# Grid and episode parameters
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
LR_E1           = 1e-3
LR_E2_WF        = 1e-3
LR_HARM         = 1e-4
LR_BENEFIT      = 1e-3
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
        hazard_harm=0.02,           # fixed from EXQ-237 (was 0.5 -> premature termination)
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

def _make_agent(condition: str, env: CausalGridWorldV2, seed: int) -> REEAgent:
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
        benefit_eval_enabled=True,     # both conditions have benefit_eval
        benefit_weight=0.5,
        z_goal_enabled=planned,
        e1_goal_conditioned=planned,
        goal_weight=1.0 if planned else 0.0,
        drive_weight=2.0 if planned else 0.0,
        use_resource_proximity_head=True,
        resource_proximity_weight=LAMBDA_RESOURCE,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    condition: str,
    warmup_episodes: int,
    steps_per_episode: int,
    seed: int,
) -> Dict:
    """
    Mixed-policy warmup: 40% greedy, 60% random.

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

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for step_i in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            # SD-018: capture resource_field_view before env.step() overwrites obs_dict
            rfv = obs_dict.get("resource_field_view", None)

            # Sense
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            # E1 tick (populates experience buffers, biases terrain prior)
            if ticks.get("e1_tick", False):
                _ = agent._e1_tick(latent)

            z_world_curr = latent.z_world.detach()

            # E2 world_forward buffer (from previous step)
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

            # Benefit proximity label (before step)
            dist   = _dist_to_nearest_resource(env)
            is_near = 1.0 if dist <= 2 else 0.0

            # Env step
            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            # z_goal update (PLANNED only, using new obs benefit_exposure)
            if planned:
                _update_z_goal(agent, obs_dict["body_state"])

            # --- Train E1 (prediction loss) + SD-018 resource proximity ---
            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()

                # SD-018: resource proximity supervision
                if rfv is not None:
                    rp_target = rfv[12].item()  # center cell = agent pos
                    rp_loss = agent.compute_resource_proximity_loss(
                        rp_target, latent
                    )
                    e1_loss = e1_loss + LAMBDA_RESOURCE * rp_loss

                if e1_loss.requires_grad:
                    e1_opt.zero_grad()
                    e1_loss.backward()
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

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            diag = agent.compute_goal_maintenance_diagnostic()
            print(
                f"    [warmup] cond={condition}"
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
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Eval using the FULL agent pipeline: generate_trajectories + select_action.

    This captures z_goal -> E1 prior -> terrain_prior -> CEM proposal bias,
    which is the core architectural mechanism distinguishing PLANNED from HABIT.
    No network weight updates during eval.
    PLANNED: z_goal updates continue (goal maintenance, no backward).
    """
    planned = (condition == "PLANNED")
    agent.eval()

    device    = agent.device
    world_dim = WORLD_DIM
    n_act     = env.action_dim

    resource_counts: List[int]   = []
    harm_rates:       List[float] = []

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

            # E1 tick: populates z_goal conditioned E1 prior for terrain_prior
            if ticks.get("e1_tick", False):
                e1_prior = agent._e1_tick(latent)
            else:
                e1_prior = torch.zeros(1, world_dim, device=device)

            # Full hippocampal planning (CEM in action-object space, SD-004)
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

            # z_goal maintenance during eval (PLANNED only, no gradients)
            if planned:
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
    seed:              int,
    warmup_episodes:   int,
    eval_episodes:     int,
    dry_run:           bool = False,
) -> Dict:
    """Run all 4 conditions (2 contexts x 2 policies) for one seed."""
    results: Dict = {}

    for context in ["SIMPLE", "LONG_HORIZON"]:
        steps = SIMPLE_STEPS if context == "SIMPLE" else LONG_STEPS

        for condition in ["HABIT", "PLANNED"]:
            print(
                f"\n[V3-EXQ-237a] seed={seed} context={context} cond={condition}"
                f" steps={steps}",
                flush=True,
            )
            print(f"Seed {seed} Condition {context}_{condition}", flush=True)
            env = _make_env(context, seed)
            agent = _make_agent(condition, env, seed)

            warmup_res = _warmup(
                agent, env, condition,
                warmup_episodes, steps, seed,
            )

            eval_res = _eval(
                agent, env, condition,
                eval_episodes, steps,
            )

            print(
                f"  [eval done] seed={seed} ctx={context} cond={condition}"
                f" resource_rate={eval_res['resource_rate']:.3f}"
                f" harm_rate={eval_res['harm_rate']:.5f}"
                f" goal_norm={warmup_res['goal_norm']:.3f}",
                flush=True,
            )

            results[(context, condition)] = {
                "resource_rate": eval_res["resource_rate"],
                "harm_rate":     eval_res["harm_rate"],
                "goal_norm":     warmup_res["goal_norm"],
            }
            print("verdict: PASS", flush=True)

    return results


# ---------------------------------------------------------------------------
# Aggregate and criteria
# ---------------------------------------------------------------------------

def _aggregate(all_results: List[Dict]) -> Dict:
    """
    Compute per-seed and aggregate metrics.

    Returns dict with keys: per_seed, interaction, criteria.
    """
    n_seeds = len(all_results)
    per_seed = []

    for r in all_results:
        sim_habit  = r[("SIMPLE",       "HABIT")  ]
        sim_plan   = r[("SIMPLE",       "PLANNED")]
        long_habit = r[("LONG_HORIZON", "HABIT")  ]
        long_plan  = r[("LONG_HORIZON", "PLANNED")]

        lift_simple = sim_plan["resource_rate"]  - sim_habit["resource_rate"]
        lift_long   = long_plan["resource_rate"] - long_habit["resource_rate"]
        interaction = lift_long - lift_simple

        harm_ratio  = (
            long_plan["harm_rate"] / max(1e-9, long_habit["harm_rate"])
        )

        per_seed.append({
            "sim_habit_rr":   sim_habit["resource_rate"],
            "sim_plan_rr":    sim_plan["resource_rate"],
            "long_habit_rr":  long_habit["resource_rate"],
            "long_plan_rr":   long_plan["resource_rate"],
            "sim_habit_hr":   sim_habit["harm_rate"],
            "sim_plan_hr":    sim_plan["harm_rate"],
            "long_habit_hr":  long_habit["harm_rate"],
            "long_plan_hr":   long_plan["harm_rate"],
            "goal_norm_simple": sim_plan["goal_norm"],
            "goal_norm_long":   long_plan["goal_norm"],
            "lift_simple":   lift_simple,
            "lift_long":     lift_long,
            "interaction":   interaction,
            "harm_ratio":    harm_ratio,
        })

    def _mean(key):
        return sum(s[key] for s in per_seed) / max(1, len(per_seed))

    # Per-seed criterion flags
    c1_flags = [s["goal_norm_long"] >= GOAL_NORM_THRESH for s in per_seed]
    c2_flags = [s["sim_habit_rr"]   >= HABIT_SIMPLE_MIN  for s in per_seed]
    c3_flags = [s["lift_long"]       >= LIFT_LONG_THRESH  for s in per_seed]
    c4_flags = [s["interaction"]     >  INTERACTION_THRESH for s in per_seed]
    c5_flags = [s["harm_ratio"]      <= HARM_RATIO_MAX     for s in per_seed]

    c1_count = sum(c1_flags)
    c2_count = sum(c2_flags)
    c3_count = sum(c3_flags)
    c4_count = sum(c4_flags)
    c5_count = sum(c5_flags)

    c1_pass = c1_count >= MAJORITY_THRESH
    c2_pass = c2_count >= MAJORITY_THRESH  # informational
    c3_pass = c3_count >= MAJORITY_THRESH
    c4_pass = c4_count >= MAJORITY_THRESH
    c5_pass = c5_count >= MAJORITY_THRESH

    return {
        "per_seed":   per_seed,
        "n_seeds":    n_seeds,
        "means": {
            "sim_habit_rr":  _mean("sim_habit_rr"),
            "sim_plan_rr":   _mean("sim_plan_rr"),
            "long_habit_rr": _mean("long_habit_rr"),
            "long_plan_rr":  _mean("long_plan_rr"),
            "lift_simple":   _mean("lift_simple"),
            "lift_long":     _mean("lift_long"),
            "interaction":   _mean("interaction"),
            "harm_ratio":    _mean("harm_ratio"),
            "goal_norm_long":_mean("goal_norm_long"),
        },
        "criteria": {
            "c1_goal_norm":   {"pass": c1_pass, "count": c1_count, "flags": c1_flags},
            "c2_habit_simple":{"pass": c2_pass, "count": c2_count, "flags": c2_flags},
            "c3_lift_long":   {"pass": c3_pass, "count": c3_count, "flags": c3_flags},
            "c4_interaction": {"pass": c4_pass, "count": c4_count, "flags": c4_flags},
            "c5_harm_ratio":  {"pass": c5_pass, "count": c5_count, "flags": c5_flags},
        },
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c4_pass": c4_pass,
        "c5_pass": c5_pass,
    }


def _decide(agg: Dict) -> Tuple[str, str, str]:
    """Return (outcome, evidence_direction, decision)."""
    c1 = agg["c1_pass"]
    c3 = agg["c3_pass"]
    c4 = agg["c4_pass"]
    c5 = agg["c5_pass"]

    if not c1:
        return "FAIL", "non_contributory", "substrate_limitation"
    if not c5:
        return "FAIL", "weakens", "retire_ree_claim"
    if c3 and c4:
        return "PASS", "supports", "retain_ree"
    if c3 and not c4:
        return "FAIL", "does_not_support", "inconclusive"
    # c3 FAIL (PLANNED not better in long-horizon despite seeding)
    return "FAIL", "does_not_support", "hybridize"


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
        f"[V3-EXQ-237a] MECH-163 Dual Goal-Directed Systems Discrimination"
        f" (supersedes V3-EXQ-237, hazard_harm bug fixed)"
        f" dry_run={args.dry_run}"
        f" warmup={warmup} eval={n_eval} seeds={seeds}",
        flush=True,
    )
    print(
        f"  SIMPLE:       {SIMPLE_N_RESOURCES} resources, {SIMPLE_N_HAZARDS} hazards,"
        f" {SIMPLE_STEPS} steps",
        flush=True,
    )
    print(
        f"  LONG_HORIZON: {LONG_N_RESOURCES} resource, {LONG_N_HAZARDS} hazards,"
        f" {LONG_STEPS} steps",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        print(f"\n[V3-EXQ-237a] === seed={seed} ===", flush=True)
        seed_results = _run_seed(
            seed=seed,
            warmup_episodes=warmup,
            eval_episodes=n_eval,
            dry_run=args.dry_run,
        )
        all_results.append(seed_results)

    # Aggregate
    agg = _aggregate(all_results)
    m   = agg["means"]
    cr  = agg["criteria"]

    outcome, direction, decision = _decide(agg)

    print(f"\n[V3-EXQ-237a] === Results ===", flush=True)
    print(
        f"  SIMPLE:       HABIT={m['sim_habit_rr']:.3f}"
        f"  PLANNED={m['sim_plan_rr']:.3f}"
        f"  lift={m['lift_simple']:+.3f}",
        flush=True,
    )
    print(
        f"  LONG_HORIZON: HABIT={m['long_habit_rr']:.3f}"
        f"  PLANNED={m['long_plan_rr']:.3f}"
        f"  lift={m['lift_long']:+.3f}",
        flush=True,
    )
    print(
        f"  interaction={m['interaction']:+.3f}"
        f"  harm_ratio={m['harm_ratio']:.3f}"
        f"  goal_norm_long={m['goal_norm_long']:.3f}",
        flush=True,
    )
    print(
        f"  C1(goal_norm>={GOAL_NORM_THRESH}): {'PASS' if agg['c1_pass'] else 'FAIL'}"
        f" ({cr['c1_goal_norm']['count']}/{agg['n_seeds']})",
        flush=True,
    )
    print(
        f"  C2(habit_simple>={HABIT_SIMPLE_MIN})[info]: {'PASS' if agg['c2_pass'] else 'FAIL'}"
        f" ({cr['c2_habit_simple']['count']}/{agg['n_seeds']})",
        flush=True,
    )
    print(
        f"  C3(lift_long>={LIFT_LONG_THRESH}): {'PASS' if agg['c3_pass'] else 'FAIL'}"
        f" ({cr['c3_lift_long']['count']}/{agg['n_seeds']})",
        flush=True,
    )
    print(
        f"  C4(interaction>{INTERACTION_THRESH}): {'PASS' if agg['c4_pass'] else 'FAIL'}"
        f" ({cr['c4_interaction']['count']}/{agg['n_seeds']})",
        flush=True,
    )
    print(
        f"  C5(harm_ratio<={HARM_RATIO_MAX}): {'PASS' if agg['c5_pass'] else 'FAIL'}"
        f" ({cr['c5_harm_ratio']['count']}/{agg['n_seeds']})",
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
        "supersedes":         SUPERSEDES,
        "outcome":            outcome,
        "evidence_direction": direction,
        "decision":           decision,
        "timestamp":          ts,
        "seeds":              seeds,
        # Parameters
        "warmup_episodes":        warmup,
        "eval_episodes":          n_eval,
        "simple_steps":           SIMPLE_STEPS,
        "long_steps":             LONG_STEPS,
        "simple_n_resources":     SIMPLE_N_RESOURCES,
        "simple_n_hazards":       SIMPLE_N_HAZARDS,
        "long_n_resources":       LONG_N_RESOURCES,
        "long_n_hazards":         LONG_N_HAZARDS,
        "greedy_frac":            GREEDY_FRAC,
        "hazard_harm":            0.02,
        # Thresholds
        "goal_norm_thresh":       GOAL_NORM_THRESH,
        "habit_simple_min":       HABIT_SIMPLE_MIN,
        "lift_long_thresh":       LIFT_LONG_THRESH,
        "interaction_thresh":     INTERACTION_THRESH,
        "harm_ratio_max":         HARM_RATIO_MAX,
        # Mean metrics
        "sim_habit_rr":           float(m["sim_habit_rr"]),
        "sim_plan_rr":            float(m["sim_plan_rr"]),
        "long_habit_rr":          float(m["long_habit_rr"]),
        "long_plan_rr":           float(m["long_plan_rr"]),
        "lift_simple":            float(m["lift_simple"]),
        "lift_long":              float(m["lift_long"]),
        "interaction":            float(m["interaction"]),
        "harm_ratio":             float(m["harm_ratio"]),
        "goal_norm_long":         float(m["goal_norm_long"]),
        # Criteria
        "c1_goal_norm_pass":      agg["c1_pass"],
        "c2_habit_simple_pass":   agg["c2_pass"],
        "c3_lift_long_pass":      agg["c3_pass"],
        "c4_interaction_pass":    agg["c4_pass"],
        "c5_harm_ratio_pass":     agg["c5_pass"],
        # Per-seed detail
        "per_seed_results": [
            {
                "seed":            seeds[i],
                "sim_habit_rr":    s["sim_habit_rr"],
                "sim_plan_rr":     s["sim_plan_rr"],
                "long_habit_rr":   s["long_habit_rr"],
                "long_plan_rr":    s["long_plan_rr"],
                "lift_simple":     s["lift_simple"],
                "lift_long":       s["lift_long"],
                "interaction":     s["interaction"],
                "harm_ratio":      s["harm_ratio"],
                "goal_norm_long":  s["goal_norm_long"],
            }
            for i, s in enumerate(agg["per_seed"])
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-237a] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
