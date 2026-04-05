#!/opt/local/bin/python3
"""
V3-EXQ-228a -- ARC-032: Theta-Rate Pathway Behavioral Test
  Supersedes V3-EXQ-228 (metric-ablation mismatch: compute_prediction_loss()
  has zero dependency on theta_buffer; ablation had no effect on measurement)

Claims: ARC-032
EXPERIMENT_PURPOSE = "evidence"

=== BUG FIX (EXQ-228 -> EXQ-228a) ===

EXQ-228 measured E1 prediction error (compute_prediction_loss()) under
THETA_BYPASS_ENABLED vs THETA_BYPASS_DISABLED conditions and found
byte-for-byte identical results across all 3 seeds.

Root cause: compute_prediction_loss() computes internal E1 MSE from
E1's own experience buffer. This has ZERO dependency on theta_buffer:
  1. EXQ-228 warmup used greedy/random actions -- generate_trajectories()
     was never called, so theta_buffer.summary() was never invoked.
  2. EXQ-228 eval also used greedy/random actions (NOT generate_trajectories).
  3. compute_prediction_loss() reads E1's internal _world_experience_buffer,
     not theta_buffer. Zeroing the theta deques had no effect at all.

Fix: switch to BEHAVIORAL OUTCOME metrics (resource_collection_rate,
harm_rate) measured via the full agent pipeline (generate_trajectories +
select_action). The ARC-032 prediction pathway is:
  E1._e1_tick -> theta_buffer.update -> theta_buffer.summary
             -> HippocampalModule.propose_trajectories (z_world_for_e3)
             -> E3 trajectory scoring and selection -> action

THETA_ZEROED patches theta_buffer.summary() to always return zeros,
so E3 proposes trajectories without any E1 world-context signal.

Both conditions are trained from scratch with an identical warmup
procedure (greedy/random, no generate_trajectories). The theta patch
has no effect during warmup (summary() is never called there).
During eval, THETA_ACTIVE gets theta-averaged z_world; THETA_ZEROED
gets zeros -- the only difference between the two trained agents.

=== SCIENTIFIC QUESTION ===

Does theta-rate packaging of E1 output (MECH-089) provide functionally
meaningful context to E3's trajectory proposals?

ARC-032 predicts that theta-averaged z_world is the primary pathway by
which E1 goal-conditioned state reaches E3. Without it (THETA_ZEROED),
E3 receives only zeros as z_world when proposing trajectories:
  - HippocampalModule CEM operates with zero z_world context
  - harm_eval and benefit_eval score all proposals identically
  - Action selection degrades toward random

Prediction: THETA_ACTIVE resource_rate > THETA_ZEROED resource_rate.

=== DESIGN ===

Factor: theta channel (BETWEEN agents, both trained from scratch)
  THETA_ACTIVE: theta_buffer.summary() -> theta-averaged z_world -> E3
                (normal operation, no patching)
  THETA_ZEROED: theta_buffer.summary() -> zeros always
                (monkey-patched before warmup starts; no effect during
                 warmup since generate_trajectories is not called;
                 takes effect during eval)

Both conditions:
  z_goal_enabled=True, e1_goal_conditioned=True, drive_weight=2.0
  benefit_eval_enabled=True, benefit_weight=0.5
  alpha_world=0.9, alpha_self=0.3

Warmup: 150 episodes, 40% greedy toward nearest resource, 60% random.
  Trains: E1 (prediction loss), E2 (world_forward), E3 (harm_eval,
  benefit_eval), z_goal attractor. Uses mixed greedy/random -- NOT
  generate_trajectories. theta_buffer is filled (via _e1_tick) but
  summary() is never consumed.

Eval: 100 episodes, full generate_trajectories() + select_action().
  z_goal updates continue during eval (no gradients).
  THETA_ACTIVE: summary() returns real theta-averaged z_world.
  THETA_ZEROED: summary() always returns zeros.

Precondition: z_goal_norm >= GOAL_NORM_THRESH for THETA_ACTIVE after
  warmup. If not met: non_contributory (substrate_limitation, not
  ARC-032 falsification -- z_goal must be seeded for E1 goal context
  to exist in the theta buffer in the first place).

=== PRE-REGISTERED CRITERIA ===

C1: resource_rate_active - resource_rate_zeroed >= LIFT_THRESH
    in >= 2/3 seeds
    (theta packaging improves resource collection -- ARC-032 main test)

C2: harm_rate_active <= harm_rate_zeroed * HARM_RATIO_MAX
    in >= 2/3 seeds
    (informational: theta does not cause harmful trajectory proposals)

PASS = C1 AND C2, majority >= 2/3 seeds each.

Evidence interpretation:
  PASS     -> supports (ARC-032: theta pathway enables E3 world context)
  C1 FAIL, precondition failed -> non_contributory (substrate_limitation)
  C1 FAIL, precondition met -> does_not_support (theta has no behavioral
    effect -- theta averaging is redundant with instantaneous z_world
    available to E3 through other channels, or E3 scoring is dominated
    by non-theta factors)
  C2 FAIL  -> weakens (theta pathway introduces harmful trajectory bias)
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
EXPERIMENT_TYPE    = "v3_exq_228a_arc032_theta_bypass_behavioral"
CLAIM_IDS          = ["ARC-032"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES         = "V3-EXQ-228"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
GOAL_NORM_THRESH   = 0.05   # Precondition: z_goal must seed in THETA_ACTIVE
LIFT_THRESH        = 0.05   # C1: resource_rate_active - resource_rate_zeroed
HARM_RATIO_MAX     = 1.5    # C2: harm_rate_active <= harm_rate_zeroed * this
MAJORITY_THRESH    = 2      # criteria met in >= MAJORITY_THRESH of 3 seeds

# ---------------------------------------------------------------------------
# Environment and training parameters
# ---------------------------------------------------------------------------
GRID_SIZE          = 10
N_RESOURCES        = 2
N_HAZARDS          = 3
HAZARD_HARM        = 0.02
PROXIMITY_HARM_SCALE     = 0.3
PROXIMITY_BENEFIT_SCALE  = 0.18
STEPS_PER_EP       = 200

WARMUP_EPISODES    = 150
EVAL_EPISODES      = 100
SEEDS              = [42, 7, 13]
GREEDY_FRAC        = 0.4

WORLD_DIM          = 32
SELF_DIM           = 16
MAX_BUF            = 4000
WF_BUF_MAX         = 2000
BATCH_SIZE         = 16

# Learning rates
LR_E1      = 1e-3
LR_E2_WF   = 1e-3
LR_HARM    = 1e-4
LR_BENEFIT = 1e-3


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
    """Update z_goal from current step benefit_exposure and drive_level."""
    b_exp = _get_benefit_exposure(obs_body)
    energy = _get_energy(obs_body)
    drive_level = max(0.0, 1.0 - energy)
    agent.update_z_goal(b_exp, drive_level=drive_level)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=N_RESOURCES,
        num_hazards=N_HAZARDS,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        hazard_harm=HAZARD_HARM,
        proximity_harm_scale=PROXIMITY_HARM_SCALE,
        proximity_benefit_scale=PROXIMITY_BENEFIT_SCALE,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        env_drift_interval=999,
        env_drift_prob=0.0,
    )


# ---------------------------------------------------------------------------
# Agent factory (both conditions: goal-conditioned)
# ---------------------------------------------------------------------------

def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
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
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Theta ablation patch
# ---------------------------------------------------------------------------

def _patch_theta_zeroed(agent: REEAgent) -> None:
    """
    Patch theta_buffer.summary() to always return zeros.

    This severs the E1 -> theta_buffer -> E3 pathway. E3's
    generate_trajectories() receives z_world=zeros instead of the
    theta-cycle-averaged z_world produced by E1.

    The patch is applied from the start (before warmup). During warmup,
    generate_trajectories() is not called, so summary() is never invoked
    and the patch has no effect on training. The patch becomes active
    during eval when generate_trajectories() fires on each e3_tick.
    """
    device    = agent.device
    world_dim = agent.config.latent.world_dim

    def _zeroed_summary():
        return torch.zeros(1, world_dim, device=device)

    agent.theta_buffer.summary = _zeroed_summary


# ---------------------------------------------------------------------------
# Warmup (same for both conditions)
# ---------------------------------------------------------------------------

def _warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    warmup_episodes: int,
    seed: int,
) -> Dict:
    """
    Mixed-policy warmup: 40% greedy toward nearest resource, 60% random.

    Does NOT call generate_trajectories. Trains:
      E1 (prediction loss), E2 (world_forward),
      E3 harm_eval (stratified), E3 benefit_eval (proximity labels),
      z_goal attractor (updated each step).

    theta_buffer.summary() is never called here, so the THETA_ZEROED
    patch has no effect on training.
    """
    agent.train()
    device = agent.device
    n_act  = env.action_dim

    e1_params    = list(agent.e1.parameters())
    e2_wf_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    e1_opt      = optim.Adam(e1_params,    lr=LR_E1)
    e2_wf_opt   = optim.Adam(e2_wf_params, lr=LR_E2_WF)
    harm_opt    = optim.Adam(agent.e3.harm_eval_head.parameters(),    lr=LR_HARM)
    benefit_opt = optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=LR_BENEFIT)

    wf_buf:       List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_pos_buf: List[torch.Tensor] = []
    harm_neg_buf: List[torch.Tensor] = []
    ben_zw_buf:   List[torch.Tensor] = []
    ben_lbl_buf:  List[float]        = []

    random.seed(seed)

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

            # Mixed policy
            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_toward_resource(env)
            else:
                action_idx = random.randint(0, n_act - 1)
            action_oh = _onehot(action_idx, n_act, device)
            agent._last_action = action_oh

            dist    = _dist_to_nearest_resource(env)
            is_near = 1.0 if dist <= 2 else 0.0

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

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
            print(
                f"    [warmup] ep {ep+1}/{warmup_episodes}"
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
    eval_episodes: int,
) -> Dict:
    """
    Eval using the full pipeline: generate_trajectories() + select_action().

    This is where the theta-channel difference between THETA_ACTIVE and
    THETA_ZEROED becomes operative:
      THETA_ACTIVE: _e3_tick -> theta_buffer.summary() -> real z_world -> E3
      THETA_ZEROED: _e3_tick -> theta_buffer.summary() -> zeros -> E3

    z_goal updates continue (no gradients).
    """
    agent.eval()
    device    = agent.device
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
    """
    Run THETA_ACTIVE and THETA_ZEROED conditions for one seed.

    Both agents are created and trained from scratch with the same warmup.
    THETA_ZEROED has summary() patched before warmup begins.
    """
    env_active = _make_env(seed)
    env_zeroed = _make_env(seed)

    agent_active = _make_agent(env_active, seed)
    agent_zeroed = _make_agent(env_zeroed, seed)
    _patch_theta_zeroed(agent_zeroed)

    results: Dict = {}

    for cond, agent, env in [
        ("THETA_ACTIVE", agent_active, env_active),
        ("THETA_ZEROED", agent_zeroed, env_zeroed),
    ]:
        print(
            f"\n[V3-EXQ-228a] seed={seed} cond={cond}"
            f" warmup={warmup_episodes} eval={eval_episodes}"
            f" steps={STEPS_PER_EP}",
            flush=True,
        )
        print(f"Seed {seed} Condition {cond}", flush=True)

        warmup_res = _warmup(agent, env, warmup_episodes, seed)

        print(
            f"  [warmup done] seed={seed} cond={cond}"
            f" goal_norm={warmup_res['goal_norm']:.3f}",
            flush=True,
        )

        eval_res = _eval(agent, env, eval_episodes)

        print(
            f"  [eval done] seed={seed} cond={cond}"
            f" resource_rate={eval_res['resource_rate']:.3f}"
            f" harm_rate={eval_res['harm_rate']:.5f}",
            flush=True,
        )

        results[cond] = {
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
    """Compute per-seed metrics, criteria, and aggregate means."""
    per_seed = []

    for r in all_results:
        active = r["THETA_ACTIVE"]
        zeroed = r["THETA_ZEROED"]

        lift        = active["resource_rate"] - zeroed["resource_rate"]
        harm_ratio  = active["harm_rate"] / max(1e-9, zeroed["harm_rate"])
        goal_norm   = active["goal_norm"]

        per_seed.append({
            "resource_rate_active": active["resource_rate"],
            "resource_rate_zeroed": zeroed["resource_rate"],
            "harm_rate_active":     active["harm_rate"],
            "harm_rate_zeroed":     zeroed["harm_rate"],
            "goal_norm_active":     goal_norm,
            "lift":                 lift,
            "harm_ratio":           harm_ratio,
        })

    def _mean(key):
        return sum(s[key] for s in per_seed) / max(1, len(per_seed))

    c1_flags = [s["lift"]       >= LIFT_THRESH       for s in per_seed]
    c2_flags = [s["harm_ratio"] <= HARM_RATIO_MAX     for s in per_seed]
    gn_flags = [s["goal_norm_active"] >= GOAL_NORM_THRESH for s in per_seed]

    c1_count = sum(c1_flags)
    c2_count = sum(c2_flags)
    gn_count = sum(gn_flags)

    c1_pass = c1_count >= MAJORITY_THRESH
    c2_pass = c2_count >= MAJORITY_THRESH
    gn_pass = gn_count >= MAJORITY_THRESH  # precondition

    return {
        "per_seed": per_seed,
        "n_seeds":  len(all_results),
        "means": {
            "resource_rate_active": _mean("resource_rate_active"),
            "resource_rate_zeroed": _mean("resource_rate_zeroed"),
            "harm_rate_active":     _mean("harm_rate_active"),
            "harm_rate_zeroed":     _mean("harm_rate_zeroed"),
            "goal_norm_active":     _mean("goal_norm_active"),
            "lift":                 _mean("lift"),
            "harm_ratio":           _mean("harm_ratio"),
        },
        "criteria": {
            "c1_lift":     {"pass": c1_pass, "count": c1_count, "flags": c1_flags},
            "c2_harm":     {"pass": c2_pass, "count": c2_count, "flags": c2_flags},
            "precond_goal":{"pass": gn_pass, "count": gn_count, "flags": gn_flags},
        },
        "c1_pass":      c1_pass,
        "c2_pass":      c2_pass,
        "precond_pass": gn_pass,
    }


def _decide(agg: Dict) -> Tuple[str, str, str]:
    """Return (outcome, evidence_direction, decision)."""
    precond = agg["precond_pass"]
    c1      = agg["c1_pass"]
    c2      = agg["c2_pass"]

    if not precond:
        return "FAIL", "non_contributory", "substrate_limitation"
    if not c2:
        return "FAIL", "weakens", "retire_ree_claim"
    if c1:
        return "PASS", "supports", "retain_ree"
    # c1 FAIL, precondition met
    return "FAIL", "does_not_support", "inconclusive"


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
        f"[V3-EXQ-228a] ARC-032 Theta-Rate Pathway Behavioral Test"
        f" (supersedes V3-EXQ-228, metric-ablation mismatch fix)"
        f" dry_run={args.dry_run}"
        f" warmup={warmup} eval={n_eval} seeds={seeds}",
        flush=True,
    )
    print(
        f"  env: {GRID_SIZE}x{GRID_SIZE}, {N_RESOURCES} resources,"
        f" {N_HAZARDS} hazards, {STEPS_PER_EP} steps"
        f" hazard_harm={HAZARD_HARM}",
        flush=True,
    )
    print(
        f"  conditions: THETA_ACTIVE (normal) vs THETA_ZEROED (summary()=zeros)",
        flush=True,
    )
    print(
        f"  lift_thresh={LIFT_THRESH}"
        f" harm_ratio_max={HARM_RATIO_MAX}"
        f" goal_norm_thresh={GOAL_NORM_THRESH}",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        print(f"\n[V3-EXQ-228a] === seed={seed} ===", flush=True)
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

    print(f"\n[V3-EXQ-228a] === Results ===", flush=True)
    print(
        f"  THETA_ACTIVE: resource_rate={m['resource_rate_active']:.3f}"
        f"  harm_rate={m['harm_rate_active']:.5f}",
        flush=True,
    )
    print(
        f"  THETA_ZEROED: resource_rate={m['resource_rate_zeroed']:.3f}"
        f"  harm_rate={m['harm_rate_zeroed']:.5f}",
        flush=True,
    )
    print(
        f"  lift={m['lift']:+.3f}"
        f"  harm_ratio={m['harm_ratio']:.3f}"
        f"  goal_norm_active={m['goal_norm_active']:.3f}",
        flush=True,
    )
    print(
        f"  Precondition (goal_norm>={GOAL_NORM_THRESH}):"
        f" {'PASS' if agg['precond_pass'] else 'FAIL'}"
        f" ({cr['precond_goal']['count']}/{agg['n_seeds']})",
        flush=True,
    )
    print(
        f"  C1 (lift>={LIFT_THRESH}):"
        f" {'PASS' if agg['c1_pass'] else 'FAIL'}"
        f" ({cr['c1_lift']['count']}/{agg['n_seeds']})",
        flush=True,
    )
    print(
        f"  C2 (harm_ratio<={HARM_RATIO_MAX})[info]:"
        f" {'PASS' if agg['c2_pass'] else 'FAIL'}"
        f" ({cr['c2_harm']['count']}/{agg['n_seeds']})",
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
        "grid_size":          GRID_SIZE,
        "n_resources":        N_RESOURCES,
        "n_hazards":          N_HAZARDS,
        "hazard_harm":        HAZARD_HARM,
        "steps_per_ep":       STEPS_PER_EP,
        "warmup_episodes":    warmup,
        "eval_episodes":      n_eval,
        "greedy_frac":        GREEDY_FRAC,
        # Thresholds
        "goal_norm_thresh":   GOAL_NORM_THRESH,
        "lift_thresh":        LIFT_THRESH,
        "harm_ratio_max":     HARM_RATIO_MAX,
        # Mean metrics
        "resource_rate_active": float(m["resource_rate_active"]),
        "resource_rate_zeroed": float(m["resource_rate_zeroed"]),
        "harm_rate_active":     float(m["harm_rate_active"]),
        "harm_rate_zeroed":     float(m["harm_rate_zeroed"]),
        "goal_norm_active":     float(m["goal_norm_active"]),
        "lift":                 float(m["lift"]),
        "harm_ratio":           float(m["harm_ratio"]),
        # Criteria
        "precond_goal_pass":    agg["precond_pass"],
        "c1_lift_pass":         agg["c1_pass"],
        "c2_harm_pass":         agg["c2_pass"],
        # Per-seed detail
        "per_seed_results": [
            {
                "seed":                  seeds[i],
                "resource_rate_active":  s["resource_rate_active"],
                "resource_rate_zeroed":  s["resource_rate_zeroed"],
                "harm_rate_active":      s["harm_rate_active"],
                "harm_rate_zeroed":      s["harm_rate_zeroed"],
                "goal_norm_active":      s["goal_norm_active"],
                "lift":                  s["lift"],
                "harm_ratio":            s["harm_ratio"],
            }
            for i, s in enumerate(agg["per_seed"])
        ],
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-228a] Output written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
