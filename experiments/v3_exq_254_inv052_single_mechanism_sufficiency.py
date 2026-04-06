#!/opt/local/bin/python3
"""
V3-EXQ-254 -- INV-052: Single Mechanism Sufficiency Test

Claims: INV-052
EXPERIMENT_PURPOSE = "evidence"

=== SCIENTIFIC QUESTION ===

Does any single serotonergic rescue mechanism (MECH-186, MECH-187, or MECH-188
in isolation) fully restore LONG_HORIZON performance to SIMPLE baseline? Or does
full recovery require all three mechanisms jointly?

INV-052 (Joint Necessity of Tonic System) predicts:
  - No single mechanism (C, D, or E) restores >= 80% of SIMPLE baseline on
    BOTH z_goal_norm AND behavioral_gap.
  - Each mechanism produces only partial rescue, confirming distinct pipeline
    stages with separate bottlenecks.
  - The combined condition (F: all three) should restore >= 80%, confirming
    that the full tonic system is necessary and sufficient together.

=== BIOLOGICAL GROUNDING ===

The three mechanisms correspond to distinct stages of the serotonergic tonic system:

  MECH-186 (terrain floor, valence_wanting_floor=0.05):
    Serotonergic benefit terrain maintenance -- prevents z_goal direction from
    collapsing to zero when benefit contacts are rare. Analog: tonic 5-HT in
    hippocampus/prefrontal maintaining a minimum reward signal gradient.

  MECH-187 (seeding gain, z_goal_seeding_gain=2.0):
    Incentive salience gain regulation -- amplifies the seeding signal when
    benefit does fire. Analog: 5-HT1B-mediated NAc disinhibition (Korte 2016),
    moderate elevation range x1.5-2.5.

  MECH-188 (PFC injection, z_goal_inject=0.3):
    Top-down PFC goal persistence -- DRN-mPFC pathway maintains z_goal norm
    floor during action selection when bottom-up seeding has failed.
    Analog: Miyazaki et al. 2020 DRN-5HT->mPFC promotes waiting under
    temporal uncertainty.

Joint necessity implies the depression attractor involves failures at all three
stages simultaneously: terrain collapse (MECH-186), gain suppression (MECH-187),
and loss of top-down persistence (MECH-188). Restoring only one stage leaves
the other two intact and cannot rescue behavior.

=== DESIGN ===

6 conditions x 3 seeds, PLANNED mode:
  (A) SIMPLE/no_interventions:         healthy baseline
  (B) LONG_HORIZON/no_interventions:   depression baseline (expected collapse)
  (C) LONG_HORIZON/terrain_floor_only: MECH-186 alone (valence_wanting_floor=0.05)
  (D) LONG_HORIZON/elevated_gain_only: MECH-187 alone (z_goal_seeding_gain=2.0)
  (E) LONG_HORIZON/goal_injection_only:MECH-188 alone (z_goal_inject=0.3)
  (F) LONG_HORIZON/all_three:          combined (all three mechanisms active)

SIMPLE:       8x8, 2 resources, 1 hazard,  80 steps/ep, hazard_harm=0.02
LONG_HORIZON: 8x8, 1 resource,  3 hazards, 150 steps/ep, hazard_harm=0.02

For each condition, run both HABIT (z_goal_enabled=False) and PLANNED
(z_goal_enabled=True) modes to compute the behavioral gap.
Exception: Condition A HABIT mode is used for baseline gap computation.

Warmup: 200 eps; Eval: 100 eps.

=== METRICS (per seed x condition) ===

  z_goal_norm:      mean from final warmup diagnostic (PLANNED mode)
  PLANNED_rr:       resource_rate in PLANNED eval
  HABIT_rr:         resource_rate in HABIT eval
  behavioral_gap:   PLANNED_rr - HABIT_rr

=== PASS CRITERIA ===

recovery_fraction (for single-mechanism conditions C, D, E):
  rf_goal_norm = z_goal_norm(X) / z_goal_norm(A)    per seed
  rf_beh_gap   = behavioral_gap(X) / behavioral_gap(A)  per seed (if A gap > 0)

PASS: In majority (>=2/3 seeds), NO single mechanism (C/D/E) reaches
  rf_goal_norm >= 0.80 AND rf_beh_gap >= 0.80 simultaneously.

(= the single-mechanism conditions each fall short of 80% recovery on at
 least one of the two metrics in the majority of seeds)

Documentation:
  combined_sufficient: True if condition F reaches >= 0.80 on both metrics
  in >= 2/3 seeds (supporting joint necessity).

FAIL: >= 1 single mechanism reaches full recovery in >= 2/3 seeds
  (would disconfirm joint necessity, suggest one stage is the bottleneck).

=== OUTPUT ===

  per_condition: {A: {per_seed: [...], mean: {...}}, B: ..., ..., F: ...}
  baseline_A: {mean_z_goal_norm, mean_behavioral_gap}
  recovery_fractions: {C: {...}, D: {...}, E: {...}, F: {...}}
    each: {mean_rf_goal_norm, mean_rf_beh_gap, n_fully_recovered_seeds}
  single_mechanism_sufficient: bool  (True = any of C/D/E fully recovers)
  combined_sufficient: bool          (True = F fully recovers)
  overall_pass: bool
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
EXPERIMENT_TYPE    = "v3_exq_254_inv052_single_mechanism_sufficiency"
CLAIM_IDS          = ["INV-052"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
RECOVERY_THRESH  = 0.80    # single-mechanism must NOT reach this on both metrics
PASS_SEED_COUNT  = 2       # majority = >= 2 out of 3 seeds

# ---------------------------------------------------------------------------
# Grid and episode parameters
# ---------------------------------------------------------------------------
GRID_SIZE = 8

# SIMPLE context (healthy baseline)
SIMPLE_N_RESOURCES  = 2
SIMPLE_N_HAZARDS    = 1
SIMPLE_STEPS_PER_EP = 80

# LONG_HORIZON context (depression attractor)
LONG_N_RESOURCES  = 1
LONG_N_HAZARDS    = 3
LONG_STEPS_PER_EP = 150

HAZARD_HARM = 0.02

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

# ---------------------------------------------------------------------------
# Condition definitions
# Each entry: (label, context, valence_wanting_floor, z_goal_seeding_gain, z_goal_inject)
# ---------------------------------------------------------------------------
CONDITIONS = [
    ("A", "SIMPLE",       0.0,  1.0, 0.0),   # healthy baseline
    ("B", "LONG_HORIZON", 0.0,  1.0, 0.0),   # depression baseline
    ("C", "LONG_HORIZON", 0.05, 1.0, 0.0),   # MECH-186 alone
    ("D", "LONG_HORIZON", 0.0,  2.0, 0.0),   # MECH-187 alone (moderate gain)
    ("E", "LONG_HORIZON", 0.0,  1.0, 0.3),   # MECH-188 alone
    ("F", "LONG_HORIZON", 0.05, 2.0, 0.3),   # all three combined
]
SINGLE_MECH_LABELS = ["C", "D", "E"]
COMBINED_LABEL     = "F"


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
    """Manhattan distance to nearest resource (999 if none)."""
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

def _make_env(context: str, seed: int) -> CausalGridWorldV2:
    """Create environment matching the named context."""
    common = dict(
        seed=seed,
        size=GRID_SIZE,
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


def _steps_for_context(context: str) -> int:
    return SIMPLE_STEPS_PER_EP if context == "SIMPLE" else LONG_STEPS_PER_EP


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(
    mode: str,
    label: str,
    context: str,
    valence_wanting_floor: float,
    z_goal_seeding_gain: float,
    z_goal_inject: float,
    env: CausalGridWorldV2,
    seed: int,
) -> REEAgent:
    """
    Create agent for a given mode/condition.

    mode: 'HABIT' (z_goal_enabled=False) or 'PLANNED' (z_goal_enabled=True)
    The three intervention parameters are only active in PLANNED mode.
    """
    planned = (mode == "PLANNED")
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
        # Intervention parameters (only meaningful in PLANNED mode)
        valence_wanting_floor=valence_wanting_floor if planned else 0.0,
        z_goal_seeding_gain=z_goal_seeding_gain if planned else 1.0,
        z_goal_inject=z_goal_inject if planned else 0.0,
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
    mode: str,
    label: str,
    warmup_episodes: int,
    steps_per_ep: int,
    seed: int,
) -> Dict:
    """
    Mixed-policy warmup (40% greedy, 60% random).

    Trains: E1, E2 world_forward, E3 harm_eval (stratified), E3 benefit_eval.
    PLANNED: also updates z_goal each step.
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

        for step_i in range(steps_per_ep):
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
            if planned:
                diag = agent.compute_goal_maintenance_diagnostic()
                goal_norm = diag["goal_norm"]
            else:
                goal_norm = 0.0
            print(
                f"    [train] cond={label} mode={mode} seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
                f" harm_pos={len(harm_pos_buf)}"
                f" harm_neg={len(harm_neg_buf)}"
                f" goal_norm={goal_norm:.3f}",
                flush=True,
            )

    if planned:
        diag_final = agent.compute_goal_maintenance_diagnostic()
        return {"goal_norm": float(diag_final["goal_norm"])}
    return {"goal_norm": 0.0}


# ---------------------------------------------------------------------------
# Eval (full agent pipeline)
# ---------------------------------------------------------------------------

def _eval(
    agent: REEAgent,
    env: CausalGridWorldV2,
    mode: str,
    label: str,
    eval_episodes: int,
    steps_per_ep: int,
    seed: int,
) -> Dict:
    """
    Eval using the full agent pipeline: generate_trajectories + select_action.

    For PLANNED conditions, the goal mechanisms (floor/gain/injection) are
    active during select_action() via the config baked into the agent.
    """
    planned = (mode == "PLANNED")
    agent.eval()

    device = agent.device
    n_act  = env.action_dim

    resource_counts: List[int]   = []
    harm_rates:      List[float] = []

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        ep_resources = 0
        ep_harm_sum  = 0.0
        ep_steps     = 0

        for _ in range(steps_per_ep):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            if ticks.get("e1_tick", False):
                e1_prior = agent._e1_tick(latent)
            else:
                e1_prior = torch.zeros(1, WORLD_DIM, device=device)

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
# Per-seed per-condition run
# ---------------------------------------------------------------------------

def _run_condition_seed(
    label:                str,
    context:              str,
    valence_wanting_floor: float,
    z_goal_seeding_gain:  float,
    z_goal_inject:        float,
    seed:                 int,
    warmup_episodes:      int,
    eval_episodes:        int,
) -> Dict:
    """Run HABIT and PLANNED for one condition x one seed."""
    steps_per_ep = _steps_for_context(context)
    results: Dict = {}

    for mode in ["HABIT", "PLANNED"]:
        print(
            f"\n[V3-EXQ-254] cond={label} mode={mode} seed={seed}"
            f" context={context} steps={steps_per_ep}"
            f" floor={valence_wanting_floor} gain={z_goal_seeding_gain}"
            f" inject={z_goal_inject}",
            flush=True,
        )
        env   = _make_env(context, seed)
        agent = _make_agent(
            mode=mode,
            label=label,
            context=context,
            valence_wanting_floor=valence_wanting_floor,
            z_goal_seeding_gain=z_goal_seeding_gain,
            z_goal_inject=z_goal_inject,
            env=env,
            seed=seed,
        )

        warmup_res = _warmup(
            agent, env, mode, label,
            warmup_episodes, steps_per_ep, seed,
        )
        eval_res = _eval(
            agent, env, mode, label,
            eval_episodes, steps_per_ep, seed,
        )

        print(
            f"  [eval done] cond={label} mode={mode} seed={seed}"
            f" resource_rate={eval_res['resource_rate']:.3f}"
            f" harm_rate={eval_res['harm_rate']:.5f}"
            f" goal_norm={warmup_res['goal_norm']:.3f}",
            flush=True,
        )
        results[mode] = {
            "resource_rate": eval_res["resource_rate"],
            "harm_rate":     eval_res["harm_rate"],
            "goal_norm":     warmup_res["goal_norm"],
        }

    gap = results["PLANNED"]["resource_rate"] - results["HABIT"]["resource_rate"]
    print(
        f"  Seed {seed}: behavioral_gap={gap:.4f}"
        f" z_goal_norm={results['PLANNED']['goal_norm']:.3f}",
        flush=True,
    )
    return {
        "HABIT":          results["HABIT"],
        "PLANNED":        results["PLANNED"],
        "behavioral_gap": gap,
        "z_goal_norm":    results["PLANNED"]["goal_norm"],
        "PLANNED_rr":     results["PLANNED"]["resource_rate"],
        "HABIT_rr":       results["HABIT"]["resource_rate"],
    }


# ---------------------------------------------------------------------------
# Aggregate results across seeds and compute recovery fractions
# ---------------------------------------------------------------------------

def _mean(lst: List[float]) -> float:
    return sum(lst) / max(1, len(lst))


def _compute_recovery_fractions(
    cond_results: Dict[str, List[Dict]],
    seeds: List[int],
) -> Dict:
    """
    For conditions C, D, E, F: compute recovery fraction vs baseline A.

    recovery_fraction per seed:
      rf_goal_norm = z_goal_norm(X) / z_goal_norm(A)   if z_goal_norm(A) > 0
      rf_beh_gap   = behavioral_gap(X) / behavioral_gap(A)  if gap(A) > 0

    fully_recovered_seed: rf_goal_norm >= 0.80 AND rf_beh_gap >= 0.80
    """
    rf_results: Dict[str, Dict] = {}

    for target_label in SINGLE_MECH_LABELS + [COMBINED_LABEL]:
        if target_label not in cond_results:
            continue

        rf_goal_norms = []
        rf_beh_gaps   = []
        fully_recovered = []

        for i, seed in enumerate(seeds):
            a_norm = cond_results["A"][i]["z_goal_norm"]
            a_gap  = cond_results["A"][i]["behavioral_gap"]
            x_norm = cond_results[target_label][i]["z_goal_norm"]
            x_gap  = cond_results[target_label][i]["behavioral_gap"]

            if a_norm > 1e-6:
                rf_n = x_norm / a_norm
            else:
                rf_n = 0.0
            if abs(a_gap) > 1e-6:
                rf_g = x_gap / a_gap
            else:
                rf_g = 1.0 if abs(x_gap) < 1e-6 else 0.0

            rf_goal_norms.append(float(rf_n))
            rf_beh_gaps.append(float(rf_g))
            fully_recovered.append(rf_n >= RECOVERY_THRESH and rf_g >= RECOVERY_THRESH)

        rf_results[target_label] = {
            "per_seed_rf_goal_norm": rf_goal_norms,
            "per_seed_rf_beh_gap":   rf_beh_gaps,
            "per_seed_fully_recovered": fully_recovered,
            "mean_rf_goal_norm":     float(_mean(rf_goal_norms)),
            "mean_rf_beh_gap":       float(_mean(rf_beh_gaps)),
            "n_fully_recovered_seeds": int(sum(fully_recovered)),
        }

    return rf_results


def _aggregate_condition(results_list: List[Dict]) -> Dict:
    """Mean metrics across seeds for one condition."""
    return {
        "mean_z_goal_norm":     float(_mean([r["z_goal_norm"]    for r in results_list])),
        "mean_behavioral_gap":  float(_mean([r["behavioral_gap"] for r in results_list])),
        "mean_PLANNED_rr":      float(_mean([r["PLANNED_rr"]     for r in results_list])),
        "mean_HABIT_rr":        float(_mean([r["HABIT_rr"]       for r in results_list])),
    }


# ---------------------------------------------------------------------------
# PASS/FAIL decision
# ---------------------------------------------------------------------------

def _decide(
    rf: Dict[str, Dict],
    seeds: List[int],
) -> Tuple[str, str, bool, bool, str]:
    """
    Returns (outcome, evidence_direction, single_mechanism_sufficient,
             combined_sufficient, decision).

    single_mechanism_sufficient: True if ANY of C/D/E fully recovers >= 2/3 seeds
    combined_sufficient:         True if F fully recovers >= 2/3 seeds
    PASS: NOT single_mechanism_sufficient (confirms joint necessity)
    """
    single_sufficient = False
    for lbl in SINGLE_MECH_LABELS:
        if lbl in rf and rf[lbl]["n_fully_recovered_seeds"] >= PASS_SEED_COUNT:
            single_sufficient = True
            break

    combined_sufficient = (
        COMBINED_LABEL in rf and
        rf[COMBINED_LABEL]["n_fully_recovered_seeds"] >= PASS_SEED_COUNT
    )

    if not single_sufficient:
        return (
            "PASS",
            "supports",
            False,
            combined_sufficient,
            "no_single_mechanism_sufficient_joint_necessity_confirmed",
        )
    else:
        return (
            "FAIL",
            "does_not_support",
            True,
            combined_sufficient,
            "single_mechanism_sufficient_joint_necessity_disconfirmed",
        )


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
        f"[V3-EXQ-254] INV-052: Single Mechanism Sufficiency Test"
        f" dry_run={args.dry_run}",
        flush=True,
    )
    print(
        f"  6 conditions x {len(seeds)} seeds x 2 modes (HABIT/PLANNED)",
        flush=True,
    )
    print(
        f"  SIMPLE:       {SIMPLE_N_RESOURCES}r/{SIMPLE_N_HAZARDS}h/{SIMPLE_STEPS_PER_EP}steps",
        flush=True,
    )
    print(
        f"  LONG_HORIZON: {LONG_N_RESOURCES}r/{LONG_N_HAZARDS}h/{LONG_STEPS_PER_EP}steps",
        flush=True,
    )
    print(
        f"  PASS: no single mechanism (C/D/E) reaches >={RECOVERY_THRESH*100:.0f}% recovery"
        f" on both metrics in >={PASS_SEED_COUNT}/{len(seeds)} seeds",
        flush=True,
    )
    print("  Conditions:", flush=True)
    for label, ctx, floor, gain, inject in CONDITIONS:
        print(
            f"    {label}: {ctx}"
            f"  floor={floor}  gain={gain}  inject={inject}",
            flush=True,
        )

    # Run all conditions x seeds
    cond_results: Dict[str, List[Dict]] = {lbl: [] for lbl, *_ in CONDITIONS}

    for label, ctx, floor, gain, inject in CONDITIONS:
        print(f"\n[V3-EXQ-254] === Condition {label} ({ctx}) ===", flush=True)
        for seed in seeds:
            print(f"\n[V3-EXQ-254] Seed {seed}", flush=True)
            res = _run_condition_seed(
                label=label,
                context=ctx,
                valence_wanting_floor=floor,
                z_goal_seeding_gain=gain,
                z_goal_inject=inject,
                seed=seed,
                warmup_episodes=warmup,
                eval_episodes=n_eval,
            )
            cond_results[label].append(res)
            # Per-seed verdict within condition
            pass_tag = "n/a"  # full verdict depends on recovery fractions
            print(
                f"verdict: seed={seed} cond={label}"
                f" z_goal_norm={res['z_goal_norm']:.3f}"
                f" behavioral_gap={res['behavioral_gap']:.4f}",
                flush=True,
            )

    # Compute recovery fractions and aggregate
    rf = _compute_recovery_fractions(cond_results, seeds)

    print("\n[V3-EXQ-254] === Recovery Fractions ===", flush=True)
    for lbl in SINGLE_MECH_LABELS + [COMBINED_LABEL]:
        if lbl not in rf:
            continue
        r = rf[lbl]
        print(
            f"  Cond {lbl}: rf_goal_norm={r['mean_rf_goal_norm']:.3f}"
            f"  rf_beh_gap={r['mean_rf_beh_gap']:.3f}"
            f"  fully_recovered_seeds={r['n_fully_recovered_seeds']}/{len(seeds)}",
            flush=True,
        )

    outcome, direction, single_suf, combined_suf, decision = _decide(rf, seeds)

    print(f"\n[V3-EXQ-254] === Final Verdict ===", flush=True)
    print(
        f"  single_mechanism_sufficient={single_suf}"
        f"  combined_sufficient={combined_suf}",
        flush=True,
    )
    print(
        f"  -> {outcome}  decision={decision}  direction={direction}",
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

    # Build per_condition summary
    per_condition: Dict = {}
    for label, ctx, floor, gain, inject in CONDITIONS:
        results_list = cond_results[label]
        per_condition[label] = {
            "context":               ctx,
            "valence_wanting_floor": floor,
            "z_goal_seeding_gain":   gain,
            "z_goal_inject":         inject,
            "per_seed": [
                {
                    "seed":           seeds[i],
                    "z_goal_norm":    float(r["z_goal_norm"]),
                    "behavioral_gap": float(r["behavioral_gap"]),
                    "PLANNED_rr":     float(r["PLANNED_rr"]),
                    "HABIT_rr":       float(r["HABIT_rr"]),
                    "harm_rate":      float(r["PLANNED"]["harm_rate"]),
                }
                for i, r in enumerate(results_list)
            ],
            **_aggregate_condition(results_list),
        }

    baseline_A = {
        "mean_z_goal_norm":    per_condition["A"]["mean_z_goal_norm"],
        "mean_behavioral_gap": per_condition["A"]["mean_behavioral_gap"],
    }

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
        "grid_size":          GRID_SIZE,
        "hazard_harm":        HAZARD_HARM,
        "simple_steps_per_ep":   SIMPLE_STEPS_PER_EP,
        "long_steps_per_ep":     LONG_STEPS_PER_EP,
        "simple_n_resources":    SIMPLE_N_RESOURCES,
        "simple_n_hazards":      SIMPLE_N_HAZARDS,
        "long_n_resources":      LONG_N_RESOURCES,
        "long_n_hazards":        LONG_N_HAZARDS,
        "recovery_thresh":    RECOVERY_THRESH,
        "pass_seed_count":    PASS_SEED_COUNT,
        # Results
        "per_condition":              per_condition,
        "baseline_A":                 baseline_A,
        "recovery_fractions":         rf,
        "single_mechanism_sufficient": single_suf,
        "combined_sufficient":         combined_suf,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-254] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
