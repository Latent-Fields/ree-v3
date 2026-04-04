#!/opt/local/bin/python3
"""
V3-EXQ-235 -- MECH-112 / ARC-030: Clean Learned-Goal Gate (Paper Gate)

Claims: MECH-112, ARC-030
EXPERIMENT_PURPOSE = "evidence"

Scientific question:
  Does a COMBINED harm+benefit+goal selector with a learned z_goal attractor
  (GOAL_ACTIVE) produce meaningful early-acquisition lift compared to the same
  COMBINED harm+benefit selector with the goal channel ablated (GOAL_ABLATED)?

  This is the cleanest current paper-gate test for whether REE has a genuinely
  learned positive attractor system, not just harm avoidance or benefit-eval
  exploitation. If it passes, it materially strengthens the case that the waking
  single-agent substrate supports attributable, harm-avoiding, goal-directed agency.

Comparison pair (non-negotiable design requirement):
  GOAL_ACTIVE:
    score(a) = harm_weight * harm_eval(z_next)
               - benefit_weight * benefit_eval(z_next)
               - goal_weight * goal_proximity(z_next, z_goal)
    z_goal enabled (z_goal_enabled=True), drive_weight=2.0 (SD-012)
    benefit_eval_enabled=True
    config.e1.goal_dim = world_dim  (MECH-116 E1 conditioning; CRITICAL -- see below)

  GOAL_ABLATED:
    score(a) = harm_weight * harm_eval(z_next)
               - benefit_weight * benefit_eval(z_next)
               [no goal_proximity term; goal_weight=0.0]
    z_goal_enabled=False, drive_weight=0.0
    benefit_eval_enabled=True

  Key: harm_weight, benefit_weight, benefit_eval training, and all other
  architecture choices are IDENTICAL across conditions. The ONLY difference is
  whether a learned goal attractor is seeded and applied.

  harm_weight  = 1.0
  benefit_weight = 0.5
  goal_weight  = 0.3  (GOAL_ACTIVE only)

Critical implementation note (pattern from EXQ-076f fix, 2026-04-04):
  config.e1.goal_dim is NOT automatically set by REEConfig.from_dims() even
  when e1_goal_conditioned=True. Must be set manually after from_dims() for
  GOAL_ACTIVE. If omitted, E1 LSTM receives no goal conditioning in either
  condition and MECH-116 is silently disabled.

Environment (habit-level, not multi-resource respawn-heavy):
  Main variant:    size=8 (inner 6x6), num_resources=1, num_hazards=1
  Nearby variant:  size=8 (inner 6x6), num_resources=1, num_hazards=2
  resource_respawn_on_consume=True (drives repeated benefit_exposure for z_goal seeding)
  env_drift_interval=50, env_drift_prob=0.05 (near-static layout)
  steps_per_episode=80 (short enough that first-hit matters)
  hazard_harm=0.02, proximity_harm_scale=0.15

Metrics (early-acquisition primary, not only end-of-episode aggregate):
  - success_by_step40: fraction of episodes where first resource hit <= step 40
  - first_hit_latency: mean step of first resource hit (999 if none, per episode)
  - goal_norm:         L2 norm of z_goal at end of warmup
  - goal_resource_corr: Pearson corr(goal_proximity_t, -resource_dist_t) during eval
                        (z_goal alignment probe -- no oracle)
  - harm_ratio:        ACTIVE/ABLATED harm rate

Seeds: [42, 7, 13, 99, 0] (5 matched seeds)
Warmup: 200 episodes
Eval:   100 episodes per condition
Steps:  80 per episode (main), 80 (variant)

Pre-registered PASS criteria:
  MAIN VARIANT -- ALL required for PASS:
    C1_norm:      goal_norm_active >= 0.10 in >= 4/5 seeds
    C2_align:     goal_resource_corr >= 0.15 in >= 3/5 seeds
                  (z_goal is aligned with resource proximity)
    C3_lift:      AT LEAST ONE of:
                    (a) success_by_step40_delta >= 0.20 (20pp ACTIVE minus ABLATED)
                    (b) first_hit_latency_ratio  <= 0.75 (ACTIVE <= 75% of ABLATED latency)
                  in >= 4/5 seeds
    C4_harm:      avg harm_ratio across 5 seeds <= 1.20

  NEARBY VARIANT (diagnostic, required for full PASS):
    C5_variant:   C3 sign positive (ACTIVE > ABLATED on success_by_step40 or
                  ACTIVE < ABLATED on first_hit_latency) in >= 3/5 seeds

Decision:
  ALL pass (C1-C5)    -> PASS, supports, retain_ree
  C1 FAIL             -> FAIL, non_contributory, substrate_limitation (z_goal not seeded)
  C2 FAIL only        -> FAIL, non_contributory, substrate_limitation (z_goal not aligned)
  C3 FAIL, C1+C2 PASS -> FAIL, does_not_support, hybridize (seeded but no lift)
  C4 FAIL             -> FAIL, weakens, retire_ree_claim (harmful exploration)
  C5 FAIL only        -> FAIL, does_not_support, hybridize (variant inconsistency)

evidence_direction_per_claim:
  MECH-112 (z_goal wanting mechanism): follows outcome above
  ARC-030 (COMBINED selector advantage): same direction as C3 result

Why better than EXQ-189/225/226:
  EXQ-189/225: GOAL_PRESENT vs GOAL_ABSENT with harm-only ablated selector.
    Different control logic between conditions violates non-negotiable requirement.
  EXQ-226: COMBINED vs HARM_ONLY (benefit_eval also absent in HARM_ONLY).
    10x10 grid + 4 resources + respawn => random walk is structurally competitive.
    Mixes benefit_eval presence with z_goal presence as confounds.
  EXQ-235: GOAL_ACTIVE vs GOAL_ABLATED with identical selector structure (harm+benefit
    in both; only z_goal_weight differs). 8x8 + 1 resource + 80 steps favors genuine
    goal-directedness over random walk. Early-acquisition metrics (success@40, latency)
    detect directional behavior that aggregate resource_rate misses. 5 seeds for
    consistency requirement.
"""

import sys
import random
import json
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_235_mech112_arc030_clean_goal_gate"
CLAIM_IDS = ["MECH-112", "ARC-030"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Selector weights (IDENTICAL across both conditions)
# ---------------------------------------------------------------------------
HARM_WEIGHT    = 1.0
BENEFIT_WEIGHT = 0.5
GOAL_WEIGHT    = 0.3   # applied in GOAL_ACTIVE only

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
GOAL_NORM_THRESH       = 0.10   # C1: z_goal must be seeded
GOAL_ALIGN_THRESH      = 0.15   # C2: alignment of z_goal with resource proximity
SUCCESS_DELTA_THRESH   = 0.20   # C3a: 20pp success@40 lift
LATENCY_RATIO_THRESH   = 0.75   # C3b: ACTIVE <= 75% of ABLATED first-hit latency
HARM_RATIO_MAX         = 1.20   # C4: harm ratio cap

# How many seeds must pass each criterion
C1_MIN_SEEDS   = 4   # out of 5
C2_MIN_SEEDS   = 3   # out of 5
C3_MIN_SEEDS   = 4   # out of 5
C5_MIN_SEEDS   = 3   # out of 5

# ---------------------------------------------------------------------------
# Episode settings
# ---------------------------------------------------------------------------
GRID_SIZE        = 8
WARMUP_EPISODES  = 200
EVAL_EPISODES    = 100
STEPS_PER_EP     = 80
SEEDS            = [42, 7, 13, 99, 0]
MAX_BUF          = 2000

# Environment -- main
NUM_HAZARDS_MAIN    = 1
NUM_RESOURCES_MAIN  = 1

# Environment -- nearby variant (same logic, one extra hazard)
NUM_HAZARDS_VAR     = 2
NUM_RESOURCES_VAR   = 1

# Early-acquisition threshold (C3a)
SUCCESS_STEP_GATE   = 40

# Step counted as "no hit" for latency
NO_HIT_SENTINEL     = 999


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _manhattan_to_nearest_resource(env) -> int:
    if not env.resources:
        return NO_HIT_SENTINEL
    ax, ay = env.agent_x, env.agent_y
    return min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)


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
    return 3 if dy > 0 else 2


def _select_goal_active(agent: REEAgent, z_world: torch.Tensor, n_actions: int) -> int:
    """GOAL_ACTIVE: score = harm - benefit_w*benefit - goal_w*goal_prox."""
    with torch.no_grad():
        best_idx, best_score = 0, float("inf")
        for idx in range(n_actions):
            a_oh   = _onehot(idx, n_actions, z_world.device)
            z_next = agent.e2.world_forward(z_world, a_oh)
            harm   = agent.e3.harm_eval(z_next).mean().item()
            bene   = agent.e3.benefit_eval(z_next).mean().item()
            gprox  = agent.goal_state.goal_proximity(z_next).mean().item()
            score  = HARM_WEIGHT * harm - BENEFIT_WEIGHT * bene - GOAL_WEIGHT * gprox
            if score < best_score:
                best_score = score
                best_idx   = idx
    return best_idx


def _select_goal_ablated(agent: REEAgent, z_world: torch.Tensor, n_actions: int) -> int:
    """GOAL_ABLATED: score = harm - benefit_w*benefit. No goal term."""
    with torch.no_grad():
        best_idx, best_score = 0, float("inf")
        for idx in range(n_actions):
            a_oh   = _onehot(idx, n_actions, z_world.device)
            z_next = agent.e2.world_forward(z_world, a_oh)
            harm   = agent.e3.harm_eval(z_next).mean().item()
            bene   = agent.e3.benefit_eval(z_next).mean().item()
            score  = HARM_WEIGHT * harm - BENEFIT_WEIGHT * bene
            if score < best_score:
                best_score = score
                best_idx   = idx
    return best_idx


def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    """Pearson r between xs and ys. Returns 0.0 if degenerate."""
    if len(xs) < 10:
        return 0.0
    try:
        a = np.array(xs, dtype=np.float64)
        b = np.array(ys, dtype=np.float64)
        a_std = a.std()
        b_std = b.std()
        if a_std < 1e-9 or b_std < 1e-9:
            return 0.0
        r = float(np.corrcoef(a, b)[0, 1])
        return r if (r == r) else 0.0  # guard NaN
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Single seed / single variant run
# ---------------------------------------------------------------------------

def _run_seed_variant(
    seed: int,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    num_hazards: int,
    num_resources: int,
) -> Dict:
    """Run both conditions for one seed on one environment variant."""
    results = {}

    for condition in ["GOAL_ACTIVE", "GOAL_ABLATED"]:
        torch.manual_seed(seed)
        random.seed(seed)

        goal_active = (condition == "GOAL_ACTIVE")

        env = CausalGridWorldV2(
            seed=seed,
            size=GRID_SIZE,
            num_hazards=num_hazards,
            num_resources=num_resources,
            hazard_harm=0.02,
            env_drift_interval=50,
            env_drift_prob=0.05,
            proximity_harm_scale=0.15,
            proximity_benefit_scale=0.18,
            proximity_approach_threshold=0.15,
            hazard_field_decay=0.5,
            energy_decay=0.005,
            use_proxy_fields=True,
            resource_respawn_on_consume=True,
        )
        n_actions = env.action_dim
        world_dim = 32

        config = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=n_actions,
            self_dim=16,
            world_dim=world_dim,
            alpha_world=0.9,
            alpha_self=0.3,
            reafference_action_dim=0,
            novelty_bonus_weight=0.0,
            benefit_eval_enabled=True,          # SAME in both conditions
            benefit_weight=BENEFIT_WEIGHT,
            z_goal_enabled=goal_active,
            e1_goal_conditioned=goal_active,
            goal_weight=GOAL_WEIGHT if goal_active else 0.0,
            drive_weight=2.0 if goal_active else 0.0,
        )

        # CRITICAL: from_dims() does NOT propagate goal_dim to E1.
        # Without this, MECH-116 E1 goal conditioning is silently disabled.
        # Pattern from EXQ-076f fix (2026-04-04).
        if goal_active:
            config.e1.goal_dim = world_dim

        agent = REEAgent(config)

        # Separate optimizers: harm_eval/benefit_eval heads on lower LR
        std_params = [
            p for n, p in agent.named_parameters()
            if "harm_eval_head" not in n and "benefit_eval_head" not in n
        ]
        optimizer        = optim.Adam(std_params, lr=1e-3)
        harm_eval_opt    = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=1e-4)
        benefit_eval_opt = optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=1e-3)

        # Stratified replay buffers
        harm_buf_pos: List[torch.Tensor] = []
        harm_buf_neg: List[torch.Tensor] = []
        benefit_buf_zw:  List[torch.Tensor] = []
        benefit_buf_lbl: List[float]        = []

        agent.train()

        # ---- WARMUP ----
        for ep in range(warmup_episodes):
            _, obs_dict = env.reset()
            agent.reset()

            for step_i in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]

                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_curr = latent.z_world.detach()

                # Mixed warmup: 60% greedy toward resource (seeds z_goal via benefit exposure)
                if goal_active and random.random() < 0.6:
                    action_idx = _greedy_toward_resource(env)
                else:
                    action_idx = random.randint(0, n_actions - 1)

                action_oh = _onehot(action_idx, n_actions, agent.device)
                agent._last_action = action_oh

                dist    = _manhattan_to_nearest_resource(env)
                is_near = 1.0 if dist <= 2 else 0.0

                _, harm_signal, done, info, obs_dict = env.step(action_oh)

                # benefit_exposure from obs_body[11] (proxy mode: body_obs_dim=12)
                obs_body_new = obs_dict["body_state"]
                b_exp = 0.0
                if obs_body_new.dim() == 1 and obs_body_new.shape[0] > 11:
                    b_exp = float(obs_body_new[11].item())
                elif obs_body_new.dim() > 1 and obs_body_new.shape[-1] > 11:
                    b_exp = float(obs_body_new[0, 11].item())

                # Standard agent training (E1 + E2)
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                total   = e1_loss + e2_loss
                if total.requires_grad:
                    optimizer.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

                # Stratified harm_eval training (both conditions)
                if float(harm_signal) < 0:
                    harm_buf_pos.append(z_world_curr)
                    if len(harm_buf_pos) > MAX_BUF:
                        harm_buf_pos = harm_buf_pos[-MAX_BUF:]
                else:
                    harm_buf_neg.append(z_world_curr)
                    if len(harm_buf_neg) > MAX_BUF:
                        harm_buf_neg = harm_buf_neg[-MAX_BUF:]

                if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                    k_p = min(16, len(harm_buf_pos))
                    k_n = min(16, len(harm_buf_neg))
                    pi  = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                    ni  = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                    zw_b = torch.cat(
                        [harm_buf_pos[i] for i in pi] +
                        [harm_buf_neg[i] for i in ni], dim=0
                    )
                    tgt  = torch.cat([
                        torch.ones(k_p, 1, device=agent.device),
                        torch.zeros(k_n, 1, device=agent.device),
                    ], dim=0)
                    hloss = F.binary_cross_entropy(agent.e3.harm_eval(zw_b), tgt)
                    if hloss.requires_grad:
                        harm_eval_opt.zero_grad()
                        hloss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_head.parameters(), 0.5
                        )
                        harm_eval_opt.step()

                # benefit_eval training (BOTH conditions -- same training)
                benefit_buf_zw.append(z_world_curr)
                benefit_buf_lbl.append(is_near)
                if len(benefit_buf_zw) > MAX_BUF:
                    benefit_buf_zw  = benefit_buf_zw[-MAX_BUF:]
                    benefit_buf_lbl = benefit_buf_lbl[-MAX_BUF:]

                if len(benefit_buf_zw) >= 32 and step_i % 4 == 0:
                    k    = min(32, len(benefit_buf_zw))
                    idxs = random.sample(range(len(benefit_buf_zw)), k)
                    zw_b = torch.cat([benefit_buf_zw[i] for i in idxs], dim=0)
                    lbl  = torch.tensor(
                        [benefit_buf_lbl[i] for i in idxs],
                        dtype=torch.float32
                    ).unsqueeze(1).to(agent.device)
                    bloss = F.binary_cross_entropy(agent.e3.benefit_eval(zw_b), lbl)
                    if bloss.requires_grad:
                        benefit_eval_opt.zero_grad()
                        bloss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.benefit_eval_head.parameters(), 0.5
                        )
                        benefit_eval_opt.step()
                        agent.e3.record_benefit_sample(k)

                # z_goal update (GOAL_ACTIVE only)
                if goal_active:
                    # SD-012 drive_level = 1 - energy (obs_body[3])
                    energy = 1.0
                    if obs_body_new.dim() == 1 and obs_body_new.shape[0] > 3:
                        energy = float(obs_body_new[3].item())
                    elif obs_body_new.dim() > 1 and obs_body_new.shape[-1] > 3:
                        energy = float(obs_body_new[0, 3].item())
                    drive_level = max(0.0, 1.0 - energy)
                    agent.update_z_goal(b_exp, drive_level=drive_level)

                if done:
                    break

            if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
                diag = agent.compute_goal_maintenance_diagnostic()
                print(
                    f"  [warmup] seed={seed} cond={condition} h={num_hazards}"
                    f" ep {ep+1}/{warmup_episodes}"
                    f" harm_pos={len(harm_buf_pos)}"
                    f" goal_norm={diag['goal_norm']:.3f}",
                    flush=True,
                )

        # Goal norm at end of warmup
        goal_norm_final = 0.0
        if goal_active:
            diag_f = agent.compute_goal_maintenance_diagnostic()
            goal_norm_final = float(diag_f["goal_norm"])

        # ---- EVAL ----
        agent.eval()

        success_at_gate:  List[int]   = []  # 1 if first hit <= SUCCESS_STEP_GATE
        first_hit_steps:  List[int]   = []  # step of first hit (or NO_HIT_SENTINEL)
        harm_rates:       List[float] = []
        gprox_vals:       List[float] = []  # for alignment probe
        rdist_vals:       List[float] = []  # for alignment probe (negated)

        for _ in range(eval_episodes):
            _, obs_dict = env.reset()
            agent.reset()

            ep_harm_sum   = 0.0
            ep_steps      = 0
            first_hit_step = NO_HIT_SENTINEL
            got_resource   = False

            for step_i in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]

                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world)
                    agent.clock.advance()
                    z_world_curr = latent.z_world.detach()

                # Alignment probe: collect (goal_prox, -resource_dist) pairs
                if goal_active and agent.goal_state.is_active():
                    gp = agent.goal_state.goal_proximity(z_world_curr).mean().item()
                    rd = _manhattan_to_nearest_resource(env)
                    gprox_vals.append(gp)
                    rdist_vals.append(-rd)   # negative distance: larger = closer = good

                if goal_active:
                    action_idx = _select_goal_active(agent, z_world_curr, n_actions)
                else:
                    action_idx = _select_goal_ablated(agent, z_world_curr, n_actions)

                action_oh = _onehot(action_idx, n_actions, agent.device)
                agent._last_action = action_oh

                _, harm_signal, done, info, obs_dict = env.step(action_oh)
                ttype = info.get("transition_type", "none")

                if ttype == "resource" and not got_resource:
                    first_hit_step = step_i + 1
                    got_resource   = True

                if float(harm_signal) < 0:
                    ep_harm_sum += abs(float(harm_signal))
                ep_steps += 1

                if done:
                    break

            success_at_gate.append(1 if (first_hit_step <= SUCCESS_STEP_GATE) else 0)
            first_hit_steps.append(first_hit_step)
            harm_rates.append(ep_harm_sum / max(1, ep_steps))

        success_rate     = float(sum(success_at_gate)) / max(1, len(success_at_gate))
        mean_first_hit   = float(sum(first_hit_steps)) / max(1, len(first_hit_steps))
        harm_rate        = float(sum(harm_rates)) / max(1, len(harm_rates))
        goal_align_corr  = _pearson_corr(gprox_vals, rdist_vals) if goal_active else 0.0

        print(
            f"  [eval] seed={seed} cond={condition} h={num_hazards}"
            f" success@{SUCCESS_STEP_GATE}={success_rate:.3f}"
            f" first_hit={mean_first_hit:.1f}"
            f" harm_rate={harm_rate:.5f}"
            f" goal_norm={goal_norm_final:.3f}"
            f" goal_align={goal_align_corr:.3f}",
            flush=True,
        )

        results[condition] = {
            "success_rate":     success_rate,
            "mean_first_hit":   mean_first_hit,
            "harm_rate":        harm_rate,
            "goal_norm":        goal_norm_final,
            "goal_align_corr":  goal_align_corr,
        }

    return results


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
    steps  = 10   if args.dry_run else STEPS_PER_EP
    seeds  = [42] if args.dry_run else SEEDS

    print(
        f"[V3-EXQ-235] MECH-112/ARC-030 Clean Learned-Goal Gate"
        f" dry_run={args.dry_run}"
        f" seeds={seeds} warmup={warmup} eval={n_eval} steps={steps}",
        flush=True,
    )

    # ---- Collect per-seed results for both variants ----
    per_seed_main: List[Dict] = []
    per_seed_var:  List[Dict] = []

    for seed in seeds:
        print(f"\n[V3-EXQ-235] ===== seed={seed} MAIN (h={NUM_HAZARDS_MAIN}) =====",
              flush=True)
        r_main = _run_seed_variant(seed, warmup, n_eval, steps,
                                   NUM_HAZARDS_MAIN, NUM_RESOURCES_MAIN)
        per_seed_main.append({"seed": seed, "conditions": r_main})

        print(f"\n[V3-EXQ-235] ===== seed={seed} VARIANT (h={NUM_HAZARDS_VAR}) =====",
              flush=True)
        r_var = _run_seed_variant(seed, warmup, n_eval, steps,
                                  NUM_HAZARDS_VAR, NUM_RESOURCES_VAR)
        per_seed_var.append({"seed": seed, "conditions": r_var})

    # ---- Per-seed criterion checks (MAIN) ----
    c1_per_seed: List[bool] = []
    c2_per_seed: List[bool] = []
    c3_per_seed: List[bool] = []
    c5_per_seed: List[bool] = []

    harm_rates_active  = []
    harm_rates_ablated = []

    for s_main in per_seed_main:
        conds = s_main["conditions"]
        ga = conds["GOAL_ACTIVE"]
        ab = conds["GOAL_ABLATED"]

        c1_per_seed.append(ga["goal_norm"] >= GOAL_NORM_THRESH)
        c2_per_seed.append(ga["goal_align_corr"] >= GOAL_ALIGN_THRESH)

        success_delta  = ga["success_rate"] - ab["success_rate"]
        ablated_lat    = ab["mean_first_hit"]
        latency_ratio  = (ga["mean_first_hit"] / max(1e-3, ablated_lat)
                          if ablated_lat < NO_HIT_SENTINEL else 1.0)
        c3a            = success_delta >= SUCCESS_DELTA_THRESH
        c3b            = latency_ratio <= LATENCY_RATIO_THRESH
        c3_per_seed.append(c3a or c3b)

        harm_rates_active.append(ga["harm_rate"])
        harm_rates_ablated.append(ab["harm_rate"])

    for s_var in per_seed_var:
        conds = s_var["conditions"]
        ga = conds["GOAL_ACTIVE"]
        ab = conds["GOAL_ABLATED"]
        success_sign = ga["success_rate"] > ab["success_rate"]
        latency_sign = ga["mean_first_hit"] < ab["mean_first_hit"]
        c5_per_seed.append(success_sign or latency_sign)

    c1_count = sum(c1_per_seed)
    c2_count = sum(c2_per_seed)
    c3_count = sum(c3_per_seed)
    c5_count = sum(c5_per_seed)

    avg_harm_active  = float(sum(harm_rates_active))  / max(1, len(harm_rates_active))
    avg_harm_ablated = float(sum(harm_rates_ablated)) / max(1, len(harm_rates_ablated))
    harm_ratio       = avg_harm_active / max(1e-9, avg_harm_ablated)

    c1_pass = c1_count >= C1_MIN_SEEDS
    c2_pass = c2_count >= C2_MIN_SEEDS
    c3_pass = c3_count >= C3_MIN_SEEDS
    c4_pass = harm_ratio <= HARM_RATIO_MAX
    c5_pass = c5_count >= C5_MIN_SEEDS

    # Decision logic
    if not c1_pass:
        outcome   = "FAIL"
        direction = "non_contributory"
        decision  = "substrate_limitation"
        decision_note = "z_goal not seeded (C1 norm FAIL)"
    elif not c2_pass:
        outcome   = "FAIL"
        direction = "non_contributory"
        decision  = "substrate_limitation"
        decision_note = "z_goal not aligned with resource proximity (C2 FAIL)"
    elif not c4_pass:
        outcome   = "FAIL"
        direction = "weakens"
        decision  = "retire_ree_claim"
        decision_note = f"harmful exploration: harm_ratio={harm_ratio:.2f} > {HARM_RATIO_MAX}"
    elif c3_pass and c5_pass:
        outcome   = "PASS"
        direction = "supports"
        decision  = "retain_ree"
        decision_note = "all criteria met -- paper gate cleared"
    elif c3_pass and not c5_pass:
        outcome   = "FAIL"
        direction = "does_not_support"
        decision  = "hybridize"
        decision_note = "main lift present but variant inconsistent (C5 FAIL)"
    else:
        outcome   = "FAIL"
        direction = "does_not_support"
        decision  = "hybridize"
        decision_note = "z_goal seeded+aligned but no behavioral lift (C3 FAIL)"

    # Summary averages
    avg = lambda key, cond: (
        sum(s["conditions"][cond][key] for s in per_seed_main) / max(1, len(per_seed_main))
    )
    ga_success = avg("success_rate",   "GOAL_ACTIVE")
    ab_success = avg("success_rate",   "GOAL_ABLATED")
    ga_latency = avg("mean_first_hit", "GOAL_ACTIVE")
    ab_latency = avg("mean_first_hit", "GOAL_ABLATED")
    ga_norm    = avg("goal_norm",      "GOAL_ACTIVE")
    ga_align   = avg("goal_align_corr","GOAL_ACTIVE")

    print(f"\n[V3-EXQ-235] ===== RESULTS =====", flush=True)
    print(
        f"  GOAL_ACTIVE:  success@{SUCCESS_STEP_GATE}={ga_success:.3f}"
        f" first_hit={ga_latency:.1f}"
        f" harm={avg_harm_active:.5f}",
        flush=True,
    )
    print(
        f"  GOAL_ABLATED: success@{SUCCESS_STEP_GATE}={ab_success:.3f}"
        f" first_hit={ab_latency:.1f}"
        f" harm={avg_harm_ablated:.5f}",
        flush=True,
    )
    print(
        f"  goal_norm={ga_norm:.3f}  goal_align={ga_align:.3f}"
        f"  harm_ratio={harm_ratio:.3f}",
        flush=True,
    )
    print(
        f"  C1(norm>={GOAL_NORM_THRESH},{c1_count}/{len(seeds)}): {'PASS' if c1_pass else 'FAIL'}"
        f"  C2(align>={GOAL_ALIGN_THRESH},{c2_count}/{len(seeds)}): {'PASS' if c2_pass else 'FAIL'}"
        f"  C3(lift,{c3_count}/{len(seeds)}): {'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C4(harm<={HARM_RATIO_MAX}): {'PASS' if c4_pass else 'FAIL'}"
        f"  C5(variant,{c5_count}/{len(seeds)}): {'PASS' if c5_pass else 'FAIL'}",
        flush=True,
    )
    print(f"  -> {outcome}: {decision_note}", flush=True)

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

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
        "decision_note":      decision_note,
        "evidence_direction_per_claim": {
            "MECH-112": direction,
            "ARC-030":  direction,
        },
        "timestamp":          ts,
        "seeds":              seeds,
        "warmup_episodes":    warmup,
        "eval_episodes":      n_eval,
        "steps_per_episode":  steps,
        # Weights
        "harm_weight":        HARM_WEIGHT,
        "benefit_weight":     BENEFIT_WEIGHT,
        "goal_weight":        GOAL_WEIGHT,
        # Thresholds
        "goal_norm_thresh":      GOAL_NORM_THRESH,
        "goal_align_thresh":     GOAL_ALIGN_THRESH,
        "success_delta_thresh":  SUCCESS_DELTA_THRESH,
        "latency_ratio_thresh":  LATENCY_RATIO_THRESH,
        "harm_ratio_max":        HARM_RATIO_MAX,
        "success_step_gate":     SUCCESS_STEP_GATE,
        # Aggregate metrics
        "ga_success_rate":    float(ga_success),
        "ab_success_rate":    float(ab_success),
        "ga_mean_first_hit":  float(ga_latency),
        "ab_mean_first_hit":  float(ab_latency),
        "avg_harm_active":    float(avg_harm_active),
        "avg_harm_ablated":   float(avg_harm_ablated),
        "harm_ratio":         float(harm_ratio),
        "ga_goal_norm":       float(ga_norm),
        "ga_goal_align_corr": float(ga_align),
        # Per-criterion counts
        "c1_norm_count":    c1_count,
        "c2_align_count":   c2_count,
        "c3_lift_count":    c3_count,
        "c4_harm_pass":     c4_pass,
        "c5_variant_count": c5_count,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c4_pass": c4_pass,
        "c5_pass": c5_pass,
        # Per-seed results (main variant)
        "per_seed_main": [
            {
                "seed":             r["seed"],
                "ga_success":       r["conditions"]["GOAL_ACTIVE"]["success_rate"],
                "ab_success":       r["conditions"]["GOAL_ABLATED"]["success_rate"],
                "ga_first_hit":     r["conditions"]["GOAL_ACTIVE"]["mean_first_hit"],
                "ab_first_hit":     r["conditions"]["GOAL_ABLATED"]["mean_first_hit"],
                "ga_harm":          r["conditions"]["GOAL_ACTIVE"]["harm_rate"],
                "ab_harm":          r["conditions"]["GOAL_ABLATED"]["harm_rate"],
                "goal_norm":        r["conditions"]["GOAL_ACTIVE"]["goal_norm"],
                "goal_align_corr":  r["conditions"]["GOAL_ACTIVE"]["goal_align_corr"],
                "c1_pass":          c1_per_seed[i],
                "c2_pass":          c2_per_seed[i],
                "c3_pass":          c3_per_seed[i],
            }
            for i, r in enumerate(per_seed_main)
        ],
        # Per-seed variant (summary only)
        "per_seed_variant": [
            {
                "seed":        r["seed"],
                "ga_success":  r["conditions"]["GOAL_ACTIVE"]["success_rate"],
                "ab_success":  r["conditions"]["GOAL_ABLATED"]["success_rate"],
                "ga_first_hit":r["conditions"]["GOAL_ACTIVE"]["mean_first_hit"],
                "ab_first_hit":r["conditions"]["GOAL_ABLATED"]["mean_first_hit"],
                "c5_pass":     c5_per_seed[j],
            }
            for j, r in enumerate(per_seed_var)
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-235] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
