#!/opt/local/bin/python3
"""
V3-EXQ-298 -- MECH-124 z_goal Salience Discriminative Pair

EXPERIMENT_PURPOSE: evidence

CLAIM UNDER TEST: MECH-124
  "When harm-trace salience dominates offline replay content, consolidation
  selectively amplifies harm predictions, progressively contracting option
  space -- a distinct failure mode from Q-021 behavioral flatness."

V3 PROXY (sleep not in V3):
  MECH-124's V4 consolidation risk depends critically on whether, at V3
  training end, z_goal salience is competitive with harm salience. EXQ-224
  (diagnostic) found mean ratio = 0.312 in the baseline condition -- harm
  dominates ~3:1. This discriminative pair tests whether the FULL protective
  stack (drive_weight=2.0 + resource_proximity_head) produces significantly
  higher z_goal/harm ratio than the degraded stack (drive_weight=0.0, no
  proximity head), directly assessing the V3 prerequisite conditions for
  avoiding MECH-124 consolidation lock-in.

EXPERIMENTAL DESIGN:
  Condition A (PRIMARY / PROTECTED):
    z_goal_enabled=True, drive_weight=2.0, use_resource_proximity_head=True
    Greedy resource approach 50% steps to ensure seeding opportunities.
    All V3 protective factors for z_goal salience enabled.
    Prediction: z_goal/harm ratio >= 0.4 by end of training.

  Condition B (ABLATION / UNPROTECTED):
    z_goal_enabled=True, drive_weight=0.0, use_resource_proximity_head=False
    Same greedy policy, same seeds.
    Drive modulation absent; z_world lacks resource proximity signal.
    Prediction: z_goal/harm ratio << 0.2 (goal not seeding reliably).

3 matched seeds across both conditions.

PRIMARY METRIC: goal_vs_harm_ratio = mean z_goal_norm / (mean harm_salience + eps)
  Measured at every CHECKPOINT_EVERY episodes, final value used for criteria.

PRE-REGISTERED CRITERIA (majority >= 2/3 seeds):
  C1 (absolute salience): PRIMARY final ratio >= 0.4
      Full protective stack makes z_goal competitive with harm salience.
  C2 (delta significance): PRIMARY final ratio - ABLATION final ratio >= 0.15
      Drive+proximity together produce meaningful salience improvement.
  C3 (consistency): seed-level PRIMARY ratio > ABLATION ratio in >= 2/3 seeds.
  C4 (data quality): goal_active_frac_PRIMARY >= 0.1 (z_goal seeding at all).

PASS: C1 AND C2 AND C3 AND C4 (majority vote across seeds where applicable)
FAIL: any criterion not met

Evidence direction:
  PASS -> supports (V3 protective stack prevents the condition that would
          trigger MECH-124 in V4; z_goal is competitive, replay need not
          be harm-dominated)
  FAIL -> weakens (z_goal salience remains below competitive threshold
          even with protective stack, MECH-124 V4 risk persists)

Usage:
  /opt/local/bin/python3 experiments/v3_exq_298_mech124_zgoal_salience_discriminative.py
  /opt/local/bin/python3 experiments/v3_exq_298_mech124_zgoal_salience_discriminative.py --dry-run
"""

import sys
import argparse
import random
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_298_mech124_zgoal_salience_discriminative"
CLAIM_IDS          = ["MECH-124"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_C1_PRIMARY_RATIO  = 0.4    # PRIMARY final ratio must reach this
THRESH_C2_DELTA          = 0.15   # PRIMARY - ABLATION delta must exceed this
THRESH_C4_GOAL_ACTIVE    = 0.1    # fraction of steps goal is active (data quality)
MAJORITY_THRESH          = 2      # >= 2 of 3 seeds

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
SEEDS             = [42, 7, 13]
TOTAL_EPISODES    = 300
CHECKPOINT_EVERY  = 50
STEPS_PER_EP      = 200
GREEDY_FRAC       = 0.5    # fraction of steps greedy toward nearest resource
WORLD_DIM         = 32
EPS               = 1e-6

# Dry-run scale
DRY_RUN_EPISODES = 4
DRY_RUN_STEPS    = 10

# Learning rates
LR_E1     = 1e-4
LR_E2     = 3e-4
LR_E3     = 1e-3
LR_LATENT = 1e-3

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
CONDITIONS = [
    ("A_PRIMARY",  True,  2.0, True),   # (label, z_goal_enabled, drive_weight, rp_head)
    ("B_ABLATION", True,  0.0, False),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_toward_resource(env: CausalGridWorldV2) -> int:
    """Return action index moving toward nearest resource (Manhattan distance).
    CausalGridWorldV2 actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1),
                               3=right(0,+1), 4=stay(0,0).
    """
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


def _get_energy(obs_body: torch.Tensor) -> float:
    flat = obs_body.flatten()
    return float(flat[3].item()) if flat.shape[0] > 3 else 1.0


def _get_benefit_exposure(obs_body: torch.Tensor) -> float:
    flat = obs_body.flatten()
    return float(flat[11].item()) if flat.shape[0] > 11 else 0.0


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.02,
        resource_benefit=0.3,
        resource_respawn_on_consume=True,
        env_drift_interval=10,
        env_drift_prob=0.05,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(
    env: CausalGridWorldV2,
    z_goal_enabled: bool,
    drive_weight: float,
    use_resource_proximity_head: bool,
    seed: int,
) -> REEAgent:
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        z_goal_enabled=z_goal_enabled,
        alpha_goal=0.05,
        decay_goal=0.005,
        benefit_eval_enabled=False,
        goal_weight=1.0,
        drive_weight=drive_weight,
        e1_goal_conditioned=True,
        use_resource_proximity_head=use_resource_proximity_head,
        resource_proximity_weight=0.5,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Single condition run
# ---------------------------------------------------------------------------

def _run_condition(
    label: str,
    seed: int,
    z_goal_enabled: bool,
    drive_weight: float,
    use_resource_proximity_head: bool,
    n_episodes: int,
    steps_per_ep: int,
) -> Dict:
    """Run one (condition, seed) cell. Returns checkpoint series and final metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    env   = _make_env(seed)
    agent = _make_agent(
        env,
        z_goal_enabled=z_goal_enabled,
        drive_weight=drive_weight,
        use_resource_proximity_head=use_resource_proximity_head,
        seed=seed,
    )
    device = agent.device

    e1_params    = list(agent.e1.parameters()) + list(agent.latent_stack.parameters())
    e2_params    = list(agent.e2.parameters())
    e3_params    = list(agent.e3.parameters())

    e1_opt = optim.Adam(e1_params, lr=LR_E1)
    e2_opt = optim.Adam(e2_params, lr=LR_E2)
    e3_opt = optim.Adam(e3_params, lr=LR_E3)

    agent.train()

    checkpoints: List[Dict] = []

    # Window accumulators (reset at each checkpoint)
    win_goal_norms:  List[float] = []
    win_harm_sals:   List[float] = []
    win_resources:   int = 0
    win_harm_events: int = 0
    win_steps:       int = 0
    win_goal_active: int = 0

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        # NOTE: do NOT reset goal_state across episodes (persistent attractor)

        for _step in range(steps_per_ep):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            z_self_prev: Optional[torch.Tensor] = None
            if agent._current_latent is not None:
                z_self_prev = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            # --- Measure z_goal salience ---
            z_goal_val = 0.0
            if z_goal_enabled and agent.goal_state is not None:
                z_goal_val = agent.goal_state.goal_norm()
                if agent.goal_state.is_active():
                    win_goal_active += 1

            # --- Measure harm salience via E3.harm_eval(z_world) ---
            harm_sal_val = 0.0
            try:
                with torch.no_grad():
                    harm_out = agent.e3.harm_eval(latent.z_world.detach())
                    harm_sal_val = (
                        harm_out.item() if harm_out.numel() == 1
                        else float(harm_out.mean().item())
                    )
            except Exception:
                pass

            win_goal_norms.append(z_goal_val)
            win_harm_sals.append(harm_sal_val)

            # --- Action selection: GREEDY_FRAC toward resource, rest random ---
            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_toward_resource(env)
            else:
                action_idx = random.randint(0, env.action_dim - 1)
            action = _onehot(action_idx, env.action_dim, device)

            if z_self_prev is not None:
                agent.record_transition(z_self_prev, action, latent.z_self.detach())

            _, reward, done, info, obs_dict = env.step(action)
            ttype      = info.get("transition_type", "none")
            harm_signal = float(reward) if float(reward) < 0 else 0.0

            if ttype == "resource":
                win_resources += 1
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                win_harm_events += 1

            # --- E1 + E2 training ---
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_e1_e2 = e1_loss + e2_loss
            if total_e1_e2.requires_grad:
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total_e1_e2.backward()
                torch.nn.utils.clip_grad_norm_(e1_params + e2_params, 1.0)
                e1_opt.step()
                e2_opt.step()

            # --- SD-018: resource proximity supervision (PRIMARY only) ---
            if use_resource_proximity_head:
                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    rp_target = rfv[12].item()  # centre cell of 5x5 view
                    rp_loss = agent.compute_resource_proximity_loss(rp_target, latent)
                    if rp_loss.requires_grad:
                        e1_opt.zero_grad()
                        (0.5 * rp_loss).backward()
                        torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
                        e1_opt.step()

            # --- E3 harm supervision ---
            if agent._current_latent is not None:
                z_world_d   = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=device
                )
                h_loss = F.mse_loss(agent.e3.harm_eval(z_world_d), harm_target)
                if h_loss.requires_grad:
                    e3_opt.zero_grad()
                    h_loss.backward()
                    torch.nn.utils.clip_grad_norm_(e3_params, 1.0)
                    e3_opt.step()

            # --- SD-012: Update z_goal from benefit signal with drive modulation ---
            if z_goal_enabled:
                benefit_exp = _get_benefit_exposure(obs_dict["body_state"].to(device))
                energy_val  = _get_energy(obs_dict["body_state"].to(device))
                drive_level = max(0.0, 1.0 - energy_val)
                agent.update_z_goal(benefit_exp, drive_level)

            agent.update_residue(harm_signal)
            win_steps += 1

            if done:
                break

        # --- Checkpoint every CHECKPOINT_EVERY episodes ---
        if (ep + 1) % CHECKPOINT_EVERY == 0:
            mean_goal  = (sum(win_goal_norms) / len(win_goal_norms)) if win_goal_norms else 0.0
            mean_harm  = (sum(win_harm_sals) / len(win_harm_sals)) if win_harm_sals else 0.0
            ratio      = mean_goal / (mean_harm + EPS)
            res_rate   = win_resources / win_steps if win_steps > 0 else 0.0
            harm_rate  = win_harm_events / win_steps if win_steps > 0 else 0.0
            goal_act   = win_goal_active / win_steps if win_steps > 0 else 0.0

            checkpoints.append({
                "episode":          ep + 1,
                "z_goal_norm":      round(mean_goal, 4),
                "harm_salience":    round(mean_harm, 4),
                "ratio":            round(ratio, 4),
                "resource_rate":    round(res_rate, 4),
                "harm_rate":        round(harm_rate, 4),
                "goal_active_frac": round(goal_act, 4),
            })

            print(
                f"  [train] cond={label} seed={seed}"
                f" ep {ep+1}/{n_episodes}"
                f"  goal={mean_goal:.3f}"
                f"  harm={mean_harm:.3f}"
                f"  ratio={ratio:.3f}"
                f"  goal_active={goal_act:.2f}",
                flush=True,
            )

            # Reset window
            win_goal_norms  = []
            win_harm_sals   = []
            win_resources   = 0
            win_harm_events = 0
            win_steps       = 0
            win_goal_active = 0

    final_ratio    = checkpoints[-1]["ratio"]            if checkpoints else 0.0
    final_goal_act = checkpoints[-1]["goal_active_frac"] if checkpoints else 0.0

    return {
        "condition":          label,
        "seed":               seed,
        "z_goal_enabled":     z_goal_enabled,
        "drive_weight":       drive_weight,
        "use_rp_head":        use_resource_proximity_head,
        "n_episodes":         n_episodes,
        "checkpoints":        checkpoints,
        "final_ratio":        round(final_ratio, 4),
        "final_goal_active":  round(final_goal_act, 4),
    }


# ---------------------------------------------------------------------------
# Per-seed: run both conditions
# ---------------------------------------------------------------------------

def _run_seed(
    seed: int,
    n_episodes: int,
    steps_per_ep: int,
) -> Dict:
    results = {}
    for label, z_goal_enabled, drive_weight, rp_head in CONDITIONS:
        print(
            f"\n[V3-EXQ-298] Seed {seed} Condition {label}"
            f"  drive_weight={drive_weight}"
            f"  rp_head={rp_head}",
            flush=True,
        )
        r = _run_condition(
            label=label,
            seed=seed,
            z_goal_enabled=z_goal_enabled,
            drive_weight=drive_weight,
            use_resource_proximity_head=rp_head,
            n_episodes=n_episodes,
            steps_per_ep=steps_per_ep,
        )
        ratio_primary = r["final_ratio"] if label == "A_PRIMARY" else None
        ratio_ablation = r["final_ratio"] if label == "B_ABLATION" else None
        print(
            f"  verdict: {'PASS' if r['final_ratio'] >= THRESH_C1_PRIMARY_RATIO else 'FAIL'}"
            f"  (final_ratio={r['final_ratio']:.4f})",
            flush=True,
        )
        results[label] = r
    return results


# ---------------------------------------------------------------------------
# Aggregate and criteria
# ---------------------------------------------------------------------------

def _aggregate(all_results: List[Dict], seeds: List[int]) -> Dict:
    n_seeds = len(all_results)

    primary_ratios  = []
    ablation_ratios = []
    deltas          = []
    goal_actives    = []
    seed_c3_pass    = []

    per_seed_results = []
    for i, res in enumerate(all_results):
        pr  = res["A_PRIMARY"]["final_ratio"]
        ab  = res["B_ABLATION"]["final_ratio"]
        gaf = res["A_PRIMARY"]["final_goal_active"]
        primary_ratios.append(pr)
        ablation_ratios.append(ab)
        deltas.append(pr - ab)
        goal_actives.append(gaf)
        seed_c3_pass.append(pr > ab)

        per_seed_results.append({
            "seed":                     seeds[i],
            "primary_final_ratio":      round(pr, 4),
            "ablation_final_ratio":     round(ab, 4),
            "delta":                    round(pr - ab, 4),
            "primary_goal_active_frac": round(gaf, 4),
            "c1_pass":                  pr >= THRESH_C1_PRIMARY_RATIO,
            "c2_pass":                  (pr - ab) >= THRESH_C2_DELTA,
            "c3_primary_gt_ablation":   pr > ab,
            "c4_pass":                  gaf >= THRESH_C4_GOAL_ACTIVE,
        })

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    mean_primary   = _mean(primary_ratios)
    mean_ablation  = _mean(ablation_ratios)
    mean_delta     = _mean(deltas)
    mean_goal_act  = _mean(goal_actives)

    c1_count = sum(1 for r in per_seed_results if r["c1_pass"])
    c2_count = sum(1 for r in per_seed_results if r["c2_pass"])
    c3_count = sum(1 for r in seed_c3_pass if r)
    c4_count = sum(1 for r in per_seed_results if r["c4_pass"])

    c1_pass = c1_count >= MAJORITY_THRESH
    c2_pass = c2_count >= MAJORITY_THRESH
    c3_pass = c3_count >= MAJORITY_THRESH
    c4_pass = c4_count >= MAJORITY_THRESH

    overall_pass = c1_pass and c2_pass and c3_pass and c4_pass

    return {
        "n_seeds":                   n_seeds,
        "mean_primary_ratio_final":  mean_primary,
        "mean_ablation_ratio_final": mean_ablation,
        "mean_delta":                mean_delta,
        "mean_goal_active_frac":     mean_goal_act,
        "c1_count":                  c1_count,
        "c2_count":                  c2_count,
        "c3_count":                  c3_count,
        "c4_count":                  c4_count,
        "c1_pass":                   c1_pass,
        "c2_pass":                   c2_pass,
        "c3_pass":                   c3_pass,
        "c4_pass":                   c4_pass,
        "overall_pass":              overall_pass,
        "per_seed_results":          per_seed_results,
    }


def _decide(agg: Dict) -> Tuple[str, str]:
    if not agg["c4_pass"]:
        return "FAIL", "non_contributory"
    if agg["overall_pass"]:
        return "PASS", "supports"
    return "FAIL", "weakens"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXQ-298 MECH-124 z_goal salience discriminative pair"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Short smoke test (4 eps x 10 steps, 1 seed, no output)")
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--steps",    type=int, default=STEPS_PER_EP)
    args = parser.parse_args()

    n_episodes   = DRY_RUN_EPISODES if args.dry_run else args.episodes
    steps_per_ep = DRY_RUN_STEPS    if args.dry_run else args.steps
    seeds        = [42]             if args.dry_run else SEEDS

    print(
        f"[V3-EXQ-298] MECH-124 z_goal salience discriminative pair"
        f"  dry_run={args.dry_run}"
        f"  episodes={n_episodes}  steps/ep={steps_per_ep}  seeds={seeds}",
        flush=True,
    )
    print(
        f"  A_PRIMARY: drive_weight=2.0 rp_head=True"
        f"  B_ABLATION: drive_weight=0.0 rp_head=False",
        flush=True,
    )
    print(
        f"  Thresholds: C1_ratio>={THRESH_C1_PRIMARY_RATIO}"
        f"  C2_delta>={THRESH_C2_DELTA}"
        f"  C4_goal_active>={THRESH_C4_GOAL_ACTIVE}"
        f"  majority={MAJORITY_THRESH}/{len(seeds)}",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        print(f"\n[V3-EXQ-298] === Seed {seed} ===", flush=True)
        seed_results = _run_seed(
            seed=seed,
            n_episodes=n_episodes,
            steps_per_ep=steps_per_ep,
        )
        all_results.append(seed_results)

    agg = _aggregate(all_results, seeds)
    outcome, direction = _decide(agg)

    print(f"\n[V3-EXQ-298] === Results ===", flush=True)
    for sr in agg["per_seed_results"]:
        print(
            f"  Seed {sr['seed']}:"
            f" primary={sr['primary_final_ratio']:.4f}"
            f" ablation={sr['ablation_final_ratio']:.4f}"
            f" delta={sr['delta']:+.4f}"
            f" goal_active={sr['primary_goal_active_frac']:.3f}"
            f"  C1={sr['c1_pass']} C2={sr['c2_pass']}"
            f" C3={sr['c3_primary_gt_ablation']} C4={sr['c4_pass']}",
            flush=True,
        )
    print(
        f"  Aggregate:"
        f" primary={agg['mean_primary_ratio_final']:.4f}"
        f" ablation={agg['mean_ablation_ratio_final']:.4f}"
        f" delta={agg['mean_delta']:+.4f}",
        flush=True,
    )
    print(
        f"  C1 ({agg['c1_count']}/{agg['n_seeds']} primary>={THRESH_C1_PRIMARY_RATIO}):"
        f" {'PASS' if agg['c1_pass'] else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C2 ({agg['c2_count']}/{agg['n_seeds']} delta>={THRESH_C2_DELTA}):"
        f" {'PASS' if agg['c2_pass'] else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C3 ({agg['c3_count']}/{agg['n_seeds']} primary>ablation):"
        f" {'PASS' if agg['c3_pass'] else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C4 ({agg['c4_count']}/{agg['n_seeds']} goal_active>={THRESH_C4_GOAL_ACTIVE}):"
        f" {'PASS' if agg['c4_pass'] else 'FAIL'}",
        flush=True,
    )
    print(f"  -> outcome={outcome} direction={direction}", flush=True)

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    # Write output
    ts_utc  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ts_unix = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts_unix}.json"

    # Flatten per-seed results for output
    per_seed_flat = []
    for i, res in enumerate(all_results):
        for label, _, _, _ in CONDITIONS:
            r = res[label]
            per_seed_flat.append({
                "seed":               seeds[i],
                "condition":          label,
                "final_ratio":        r["final_ratio"],
                "final_goal_active":  r["final_goal_active"],
                "n_checkpoints":      len(r["checkpoints"]),
                "checkpoints":        r["checkpoints"],
            })

    manifest = {
        "run_id":                f"{EXPERIMENT_TYPE}_{ts_unix}_v3",
        "experiment_type":       EXPERIMENT_TYPE,
        "architecture_epoch":    "ree_hybrid_guardrails_v1",
        "claim_ids":             CLAIM_IDS,
        "experiment_purpose":    EXPERIMENT_PURPOSE,
        "outcome":               outcome,
        "evidence_direction":    direction,
        "timestamp_utc":         ts_utc,
        "timestamp":             ts_unix,
        "seeds":                 seeds,
        # Parameters
        "total_episodes":            TOTAL_EPISODES,
        "checkpoint_every":          CHECKPOINT_EVERY,
        "steps_per_ep":              STEPS_PER_EP,
        "greedy_frac":               GREEDY_FRAC,
        "world_dim":                 WORLD_DIM,
        # Pre-registered thresholds
        "registered_thresholds": {
            "c1_primary_ratio_min":   THRESH_C1_PRIMARY_RATIO,
            "c2_delta_min":           THRESH_C2_DELTA,
            "c4_goal_active_min":     THRESH_C4_GOAL_ACTIVE,
            "majority_thresh":        MAJORITY_THRESH,
        },
        # Aggregate
        "primary_metric_primary_mean":  agg["mean_primary_ratio_final"],
        "primary_metric_ablation_mean": agg["mean_ablation_ratio_final"],
        "delta":                        agg["mean_delta"],
        "mean_goal_active_frac":        agg["mean_goal_active_frac"],
        "c1_pass":                      agg["c1_pass"],
        "c1_count":                     agg["c1_count"],
        "c2_pass":                      agg["c2_pass"],
        "c2_count":                     agg["c2_count"],
        "c3_pass":                      agg["c3_pass"],
        "c3_count":                     agg["c3_count"],
        "c4_pass":                      agg["c4_pass"],
        "c4_count":                     agg["c4_count"],
        "n_seeds":                      agg["n_seeds"],
        # Per-seed
        "per_seed_results":             agg["per_seed_results"],
        "per_seed_detail":              per_seed_flat,
    }

    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[V3-EXQ-298] Output written: {out_path}", flush=True)
    print(f"verdict: {'PASS' if outcome == 'PASS' else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
