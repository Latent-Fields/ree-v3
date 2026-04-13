#!/opt/local/bin/python3
"""
V3-EXQ-406 -- INV-053: Depression Attractor Replication (5-seed systematic study)

Claims: INV-053
Dispatch mode: discriminative_pair (HABIT vs PLANNED in LONG_HORIZON only)

Context:
  EXQ-237a seed=42 showed an incidental observation: in the LONG_HORIZON condition,
  both HABIT and PLANNED conditions were behaviourally indistinguishable (goal_norm
  failed to seed, both modes accumulated comparable harm, zero resource acquisition
  differential). This pattern was labelled a "depression attractor": high harm
  exposure -> z_harm_a elevation -> z_goal seeding suppressed -> no terrain building
  -> benefit evaluation remains noise -> PLANNED collapses to HABIT.

  EXQ-237a confirmed this was not universal (seeds 7 and 13 passed C1 goal_norm),
  but the seed=42 pattern establishes the attractor exists as a reachable state.

  INV-053 predicts this is robust under appropriate conditions: with enough hazard
  density and resource sparsity (LONG_HORIZON parameters), a large fraction of seeds
  will converge to the attractor state.

Design:
  LONG_HORIZON context only (8x8, 1 resource, 3 hazards, hazard_harm=0.02, 150 steps).
  Run both HABIT and PLANNED conditions per seed.
  5 seeds: [42, 7, 13, 99, 17].

  For each seed, PLANNED condition is the indicator:
    - goal_norm_long: z_goal norm at warmup end (< 0.1 = attractor state)
    - harm_rate: mean harm per step during eval
    - resource_rate: fraction of eval episodes collecting >=1 resource
    - behavioral_equivalence: |HABIT_harm_rate - PLANNED_harm_rate| (< 0.02 = attractor)
    - accumulated_harm_mean: mean cumulative harm per episode (proxy for z_harm_a elevation)

PRE-REGISTERED ACCEPTANCE CRITERIA:

  PASS (attractor confirmed): >= 4/5 seeds show A_pass, where
    A_pass = (goal_norm_long < 0.10) AND (behavioral_equivalence < 0.02)
    (z_goal failed to seed AND HABIT and PLANNED are behaviourally indistinguishable)

  PARTIAL (attractor present but minority): 2-3/5 seeds show A_pass
    -> inconclusive: attractor exists but not dominant under these parameters

  FAIL (attractor absent): <= 1/5 seeds show A_pass
    -> does_not_support: LONG_HORIZON parameters not sufficient to reliably induce attractor

Decision scoring:
  PASS (>= 4/5 seeds):     supports -- depression attractor is a robust emergent state
  PARTIAL (2-3/5 seeds):   inconclusive -- attractor reachable but not dominant
  FAIL (<= 1/5 seeds):     does_not_support -- EXQ-237a seed=42 was a spurious observation

Secondary diagnostic metrics (not criteria, for characterisation):
  - accumulated_harm_mean per seed (z_harm_a elevation proxy)
  - goal_norm trajectory across warmup epochs (qualitative)
  - PLANNED resource_rate vs HABIT resource_rate per seed
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
EXPERIMENT_TYPE    = "v3_exq_406_inv053_depression_attractor_replication"
CLAIM_IDS          = ["INV-053"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
GOAL_NORM_ATTRACTOR   = 0.10   # below = z_goal failed to seed (attractor condition)
EQUIV_THRESH          = 0.02   # |HABIT_harm_rate - PLANNED_harm_rate| < this = attractor
ATTRACTOR_SEED_THRESH = 4      # >= this many seeds must show attractor for PASS

# ---------------------------------------------------------------------------
# Grid and episode parameters (LONG_HORIZON only, matches EXQ-237a)
# ---------------------------------------------------------------------------
GRID_SIZE        = 8
LONG_N_RESOURCES = 1
LONG_N_HAZARDS   = 3
LONG_STEPS       = 150

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
WARMUP_EPISODES = 200
EVAL_EPISODES   = 100
SEEDS           = [42, 7, 13, 99, 17]
GREEDY_FRAC     = 0.4
MAX_BUF         = 4000
WF_BUF_MAX      = 2000
WORLD_DIM       = 32
BATCH_SIZE      = 16

# Learning rates (identical to EXQ-237a)
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


def _greedy_toward_resource(env: CausalGridWorldV2) -> int:
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


def _dist_to_nearest_resource(env: CausalGridWorldV2) -> int:
    if not env.resources:
        return 999
    ax, ay = env.agent_x, env.agent_y
    return min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)


def _get_benefit_exposure(obs_body: torch.Tensor) -> float:
    flat = obs_body.flatten()
    if flat.shape[0] > 11:
        return float(flat[11].item())
    return 0.0


def _get_energy(obs_body: torch.Tensor) -> float:
    flat = obs_body.flatten()
    if flat.shape[0] > 3:
        return float(flat[3].item())
    return 1.0


def _update_z_goal(agent: REEAgent, obs_body: torch.Tensor) -> None:
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
        num_resources=LONG_N_RESOURCES,
        num_hazards=LONG_N_HAZARDS,
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
    seed: int,
) -> Dict:
    """Mixed-policy warmup: 40% greedy, 60% random."""
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

    # Track accumulated harm per episode (z_harm_a elevation proxy)
    accum_harm_per_ep: List[float] = []

    random.seed(seed)
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        ep_accum_harm = 0.0

        for step_i in range(LONG_STEPS):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)
            rfv       = obs_dict.get("resource_field_view", None)

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

            ep_accum_harm += max(0.0, -float(harm_signal))

            if planned:
                _update_z_goal(agent, obs_dict["body_state"])

            # Train E1
            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if rfv is not None:
                    rp_target = rfv[12].item()
                    rp_loss = agent.compute_resource_proximity_loss(rp_target, latent)
                    e1_loss = e1_loss + LAMBDA_RESOURCE * rp_loss
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

            # Train E3 harm_eval
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

        accum_harm_per_ep.append(ep_accum_harm)

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            diag = agent.compute_goal_maintenance_diagnostic()
            mean_harm = sum(accum_harm_per_ep[-50:]) / max(1, len(accum_harm_per_ep[-50:]))
            print(
                f"    [warmup] cond={condition}"
                f" ep {ep+1}/{warmup_episodes}"
                f" harm_pos={len(harm_pos_buf)}"
                f" harm_neg={len(harm_neg_buf)}"
                f" goal_norm={diag['goal_norm']:.3f}"
                f" harm_mean_50ep={mean_harm:.5f}",
                flush=True,
            )

    diag_final = agent.compute_goal_maintenance_diagnostic()
    accumulated_harm_mean = (
        sum(accum_harm_per_ep) / max(1, len(accum_harm_per_ep))
    )
    return {
        "goal_norm": float(diag_final["goal_norm"]),
        "accumulated_harm_mean": accumulated_harm_mean,
    }


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def _eval(
    agent: REEAgent,
    env: CausalGridWorldV2,
    condition: str,
    eval_episodes: int,
) -> Dict:
    """Full agent pipeline eval. No weight updates."""
    planned = (condition == "PLANNED")
    agent.eval()

    device    = agent.device
    n_act     = env.action_dim

    resource_counts: List[int]   = []
    harm_rates:       List[float] = []

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        ep_resources = 0
        ep_harm_sum  = 0.0
        ep_steps     = 0

        for _ in range(LONG_STEPS):
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

    return {
        "resource_rate": sum(resource_counts) / max(1, len(resource_counts)),
        "harm_rate":     sum(harm_rates)      / max(1, len(harm_rates)),
    }


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def _run_seed(seed: int, warmup_episodes: int, eval_episodes: int) -> Dict:
    results: Dict = {}

    for condition in ["HABIT", "PLANNED"]:
        print(
            f"\n[V3-EXQ-406] seed={seed} cond={condition}"
            f" LONG_HORIZON n_haz={LONG_N_HAZARDS} n_res={LONG_N_RESOURCES}",
            flush=True,
        )
        env   = _make_env(seed)
        agent = _make_agent(condition, env, seed)

        warmup_res = _warmup(agent, env, condition, warmup_episodes, seed)
        eval_res   = _eval(agent, env, condition, eval_episodes)

        print(
            f"  [done] seed={seed} cond={condition}"
            f" resource_rate={eval_res['resource_rate']:.3f}"
            f" harm_rate={eval_res['harm_rate']:.5f}"
            f" goal_norm={warmup_res['goal_norm']:.3f}"
            f" accum_harm_mean={warmup_res['accumulated_harm_mean']:.5f}",
            flush=True,
        )
        results[condition] = {
            "resource_rate":         eval_res["resource_rate"],
            "harm_rate":             eval_res["harm_rate"],
            "goal_norm":             warmup_res["goal_norm"],
            "accumulated_harm_mean": warmup_res["accumulated_harm_mean"],
        }

    return results


# ---------------------------------------------------------------------------
# Aggregate and criteria
# ---------------------------------------------------------------------------

def _aggregate(all_results: List[Dict], seeds: List[int]) -> Dict:
    per_seed = []

    for seed, r in zip(seeds, all_results):
        habit   = r["HABIT"]
        planned = r["PLANNED"]

        behavioral_equiv = abs(habit["harm_rate"] - planned["harm_rate"])
        a_pass = (
            planned["goal_norm"] < GOAL_NORM_ATTRACTOR and
            behavioral_equiv < EQUIV_THRESH
        )

        per_seed.append({
            "seed":                   seed,
            "habit_resource_rate":    habit["resource_rate"],
            "planned_resource_rate":  planned["resource_rate"],
            "habit_harm_rate":        habit["harm_rate"],
            "planned_harm_rate":      planned["harm_rate"],
            "goal_norm":              planned["goal_norm"],
            "accumulated_harm_mean":  planned["accumulated_harm_mean"],
            "behavioral_equivalence": behavioral_equiv,
            "attractor_pass":         a_pass,
        })

    n_attractor = sum(1 for s in per_seed if s["attractor_pass"])

    return {
        "per_seed":     per_seed,
        "n_seeds":      len(seeds),
        "n_attractor":  n_attractor,
        "mean_goal_norm":     sum(s["goal_norm"] for s in per_seed) / max(1, len(per_seed)),
        "mean_harm_rate_habit":   sum(s["habit_harm_rate"] for s in per_seed) / max(1, len(per_seed)),
        "mean_harm_rate_planned": sum(s["planned_harm_rate"] for s in per_seed) / max(1, len(per_seed)),
        "mean_accum_harm":    sum(s["accumulated_harm_mean"] for s in per_seed) / max(1, len(per_seed)),
        "mean_behav_equiv":   sum(s["behavioral_equivalence"] for s in per_seed) / max(1, len(per_seed)),
    }


def _decide(agg: Dict) -> Tuple[str, str, str]:
    n = agg["n_attractor"]
    if n >= ATTRACTOR_SEED_THRESH:   # >= 4/5
        return "PASS",  "supports",          "retain_ree"
    if n >= 2:                        # 2-3/5
        return "FAIL",  "inconclusive",      "inconclusive"
    # <= 1/5
    return "FAIL", "does_not_support", "retire_ree_claim"


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
        f"[V3-EXQ-406] INV-053: Depression Attractor Replication"
        f" dry_run={args.dry_run}"
        f" warmup={warmup} eval={n_eval} seeds={seeds}",
        flush=True,
    )
    print(
        f"  LONG_HORIZON: {LONG_N_RESOURCES} resource, {LONG_N_HAZARDS} hazards,"
        f" {LONG_STEPS} steps/ep hazard_harm=0.02",
        flush=True,
    )
    print(
        f"  Attractor criteria: goal_norm < {GOAL_NORM_ATTRACTOR}"
        f" AND behavioral_equiv < {EQUIV_THRESH}"
        f" in >= {ATTRACTOR_SEED_THRESH}/{len(seeds)} seeds",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        print(f"\n[V3-EXQ-406] === seed={seed} ===", flush=True)
        seed_results = _run_seed(
            seed=seed,
            warmup_episodes=warmup,
            eval_episodes=n_eval,
        )
        all_results.append(seed_results)

    agg = _aggregate(all_results, seeds)
    outcome, direction, decision = _decide(agg)

    print(f"\n[V3-EXQ-406] === Aggregate Results ===", flush=True)
    for s in agg["per_seed"]:
        print(
            f"  seed={s['seed']}"
            f" goal_norm={s['goal_norm']:.3f}"
            f" behav_equiv={s['behavioral_equivalence']:.5f}"
            f" accum_harm={s['accumulated_harm_mean']:.5f}"
            f" -> attractor={'PASS' if s['attractor_pass'] else 'FAIL'}",
            flush=True,
        )
    print(
        f"  n_attractor={agg['n_attractor']}/{agg['n_seeds']}"
        f" mean_goal_norm={agg['mean_goal_norm']:.3f}"
        f" mean_behav_equiv={agg['mean_behav_equiv']:.5f}",
        flush=True,
    )
    print(f"  -> {outcome} decision={decision} direction={direction}", flush=True)

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    ts      = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    manifest = {
        "run_id":              f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":     EXPERIMENT_TYPE,
        "architecture_epoch":  "ree_hybrid_guardrails_v1",
        "claim_ids":           CLAIM_IDS,
        "experiment_purpose":  EXPERIMENT_PURPOSE,
        "outcome":             outcome,
        "evidence_direction":  direction,
        "decision":            decision,
        "timestamp":           ts,
        "seeds":               seeds,
        # Parameters
        "warmup_episodes":         warmup,
        "eval_episodes":           n_eval,
        "long_steps":              LONG_STEPS,
        "long_n_resources":        LONG_N_RESOURCES,
        "long_n_hazards":          LONG_N_HAZARDS,
        "hazard_harm":             0.02,
        "greedy_frac":             GREEDY_FRAC,
        # Thresholds
        "goal_norm_attractor":     GOAL_NORM_ATTRACTOR,
        "equiv_thresh":            EQUIV_THRESH,
        "attractor_seed_thresh":   ATTRACTOR_SEED_THRESH,
        # Aggregate metrics
        "n_attractor":             agg["n_attractor"],
        "n_seeds":                 agg["n_seeds"],
        "mean_goal_norm":          agg["mean_goal_norm"],
        "mean_harm_rate_habit":    agg["mean_harm_rate_habit"],
        "mean_harm_rate_planned":  agg["mean_harm_rate_planned"],
        "mean_accum_harm":         agg["mean_accum_harm"],
        "mean_behav_equiv":        agg["mean_behav_equiv"],
        # Per-seed detail
        "per_seed_results":        agg["per_seed"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-406] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
