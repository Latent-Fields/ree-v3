#!/opt/local/bin/python3
"""
V3-EXQ-224 -- MECH-124 z_goal Salience Diagnostic

EXPERIMENT_PURPOSE: diagnostic

MECH-124: Consolidation-Mediated Option-Space Contraction (V4 failure mode).
Risk: V4 consolidation amplifies whatever imbalance exists between z_goal salience
and harm salience at the end of V3 training. If harm dominates the residue field
while z_goal signal is weak, SWS replay will strengthen harm representations
further, progressively narrowing the option space the agent considers viable.

This experiment measures whether z_goal salience stays competitive with harm
salience over a medium-length run (~300 episodes). It is a DIAGNOSTIC, not
evidence -- results flag V4 risk if harm dominates, but do not confirm or deny
MECH-124 as a claim (that requires V4 consolidation experiments).

SD-012 (drive_weight=2.0) is now implemented: z_goal seeding should work when
benefit_exposure fires with sufficient drive_level. This experiment tests whether,
under realistic training conditions with resource-contact seeding, z_goal norm
remains competitive with harm salience.

Two conditions per seed:
  BASELINE -- z_goal_enabled=True, drive_weight=2.0, goal_weight=1.0
              50% greedy toward resource to ensure seeding
  ABLATION -- z_goal_enabled=False (pure harm, no goal signal)
              Same greedy policy, same seeds -- isolates goal contribution

At every CHECKPOINT_EVERY episodes, record:
  z_goal_norm     -- mean agent.goal_state.goal_norm() per step
  harm_salience   -- mean E3.harm_eval(z_world) per step
  resource_rate   -- resource contacts / steps
  harm_rate       -- harm events / steps
  ratio           -- z_goal_norm / (harm_salience + eps)
  goal_active_frac -- fraction of steps where goal_state.is_active()

Diagnostic flags (NOT pass/fail criteria -- this is a diagnostic):
  FLAG_1: ratio (baseline) < 0.2 at final checkpoint
           => z_goal not seeding / harm dominating => V4 risk
  FLAG_2: ratio (baseline) declining monotonically (slope < -0.002/episode)
           => z_goal salience eroding over training => V4 risk
  FLAG_3: goal_active_frac < 0.3 at final checkpoint
           => goal state rarely active => seeding problem persists
  RISK_DETECTED: FLAG_1 or FLAG_2 (even one is enough to flag for V4 design)

If RISK_DETECTED: recommend adding balanced replay scheduling (MECH-121 guard)
to V4 design before consolidation experiments begin.

Usage:
  /opt/local/bin/python3 experiments/v3_exq_224_mech124_zgoal_salience_diag.py
  /opt/local/bin/python3 experiments/v3_exq_224_mech124_zgoal_salience_diag.py --dry-run
"""

import sys
import argparse
import random
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# -----------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_224_mech124_zgoal_salience_diag"
CLAIM_IDS = ["MECH-124"]
EXPERIMENT_PURPOSE = "diagnostic"

# Training parameters
DEFAULT_EPISODES   = 300
CHECKPOINT_EVERY   = 50    # record metrics every N episodes
STEPS_PER_EP       = 200
GREEDY_FRAC        = 0.5   # fraction of steps using greedy resource-approach
SEEDS              = [42, 7, 13]
EPS                = 1e-6  # prevent division by zero in ratio

# Dry-run scale
DRY_RUN_EPISODES   = 4
DRY_RUN_STEPS      = 10
# -----------------------------------------------------------------------


def _greedy_action_toward_resource(env) -> int:
    """Return action index moving toward nearest resource (L1 distance).
    Falls back to random if no resources.
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
    dx = rx - ax
    dy = ry - ay
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0
    else:
        return 3 if dy > 0 else 2


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


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


def _make_agent(
    env: CausalGridWorldV2,
    z_goal_enabled: bool,
    world_dim: int = 32,
) -> REEAgent:
    """Create REE agent for MECH-124 diagnostic conditions."""
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        z_goal_enabled=z_goal_enabled,
        alpha_goal=0.05,
        decay_goal=0.005,
        benefit_eval_enabled=False,
        goal_weight=1.0,
        drive_weight=2.0,   # SD-012: homeostatic drive amplification
        e1_goal_conditioned=True,
    )
    return REEAgent(config)


def _run_condition(
    label: str,
    seed: int,
    n_episodes: int,
    steps_per_ep: int,
    z_goal_enabled: bool,
    world_dim: int = 32,
) -> Dict:
    """Run one condition; return checkpoint series and final metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    env   = _make_env(seed)
    agent = _make_agent(env, z_goal_enabled=z_goal_enabled, world_dim=world_dim)
    device = agent.device

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-4)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-4)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=1e-3,
    )

    agent.train()

    checkpoints: List[Dict] = []

    # Window accumulators (reset after each checkpoint)
    win_goal_norms:  List[float] = []
    win_harm_sals:   List[float] = []
    win_resources:   int = 0
    win_harm_events: int = 0
    win_steps:       int = 0
    win_goal_active: int = 0  # steps where goal_state.is_active()

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        # Note: do NOT reset goal_state across episodes (persistent attractor)

        for _ in range(steps_per_ep):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            # Capture z_self before sense() for E2 transition recording
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

            # --- Action selection: 50% greedy toward resource, 50% random ---
            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_action_toward_resource(env)
            else:
                action_idx = random.randint(0, env.action_dim - 1)
            action = _onehot(action_idx, env.action_dim, device)

            if z_self_prev is not None:
                agent.record_transition(z_self_prev, action, latent.z_self.detach())

            _, reward, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            harm_signal = float(reward) if float(reward) < 0 else 0.0

            if ttype == "resource":
                win_resources += 1
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                win_harm_events += 1

            # --- Training ---
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_e1_e2 = e1_loss + e2_loss
            if total_e1_e2.requires_grad:
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total_e1_e2.backward()
                e1_opt.step()
                e2_opt.step()

            # E3 harm supervision
            if agent._current_latent is not None:
                z_world_d = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=device
                )
                h_loss = F.mse_loss(agent.e3.harm_eval(z_world_d), harm_target)
                e3_opt.zero_grad()
                h_loss.backward()
                e3_opt.step()

            # SD-012: Update z_goal from benefit signal with drive modulation
            if z_goal_enabled:
                benefit_exp = float(obs_body[0, 11]) if obs_body.dim() == 2 else float(obs_body[11])
                energy_val  = float(obs_body[0, 3])  if obs_body.dim() == 2 else float(obs_body[3])
                drive_level = 1.0 - energy_val
                agent.update_z_goal(benefit_exp, drive_level)

            agent.update_residue(harm_signal)
            win_steps += 1

            if done:
                break

        # --- Checkpoint ---
        if (ep + 1) % CHECKPOINT_EVERY == 0:
            mean_goal = (
                sum(win_goal_norms) / len(win_goal_norms) if win_goal_norms else 0.0
            )
            mean_harm = (
                sum(win_harm_sals) / len(win_harm_sals) if win_harm_sals else 0.0
            )
            ratio     = mean_goal / (mean_harm + EPS)
            res_rate  = win_resources  / win_steps if win_steps > 0 else 0.0
            harm_rate = win_harm_events / win_steps if win_steps > 0 else 0.0
            goal_act  = win_goal_active / win_steps if win_steps > 0 else 0.0

            checkpoints.append({
                "episode":         ep + 1,
                "z_goal_norm":     round(mean_goal, 4),
                "harm_salience":   round(mean_harm, 4),
                "ratio":           round(ratio, 4),
                "resource_rate":   round(res_rate, 4),
                "harm_rate":       round(harm_rate, 4),
                "goal_active_frac": round(goal_act, 4),
            })

            print(
                f"  [{label}] ep {ep+1:>4}"
                f"  goal={mean_goal:.3f}"
                f"  harm_sal={mean_harm:.3f}"
                f"  ratio={ratio:.3f}"
                f"  res={res_rate:.3f}"
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

    # --- Compute ratio slope (linear regression over checkpoint series) ---
    ratios = [c["ratio"] for c in checkpoints]
    n_pts  = len(ratios)
    slope  = 0.0
    if n_pts >= 2:
        x_mean = (n_pts - 1) / 2.0
        y_mean = sum(ratios) / n_pts
        num    = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(ratios))
        denom  = sum((i - x_mean) ** 2 for i in range(n_pts))
        slope  = num / denom if denom != 0 else 0.0

    final_ratio      = ratios[-1] if ratios else 0.0
    final_goal_act   = checkpoints[-1]["goal_active_frac"] if checkpoints else 0.0
    final_res_rate   = checkpoints[-1]["resource_rate"]    if checkpoints else 0.0
    final_harm_rate  = checkpoints[-1]["harm_rate"]        if checkpoints else 0.0

    return {
        "label":             label,
        "z_goal_enabled":    z_goal_enabled,
        "n_episodes":        n_episodes,
        "world_dim":         32,
        "checkpoints":       checkpoints,
        "final_ratio":       round(final_ratio, 4),
        "ratio_slope":       round(slope, 6),
        "final_goal_active_frac": round(final_goal_act, 4),
        "final_resource_rate":    round(final_res_rate, 4),
        "final_harm_rate":        round(final_harm_rate, 4),
    }


def run(
    n_episodes: int = DEFAULT_EPISODES,
    steps_per_ep: int = STEPS_PER_EP,
    seeds: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict:
    """Run the full MECH-124 salience diagnostic and write result JSON."""
    if seeds is None:
        seeds = SEEDS
    if dry_run:
        n_episodes = DRY_RUN_EPISODES
        steps_per_ep = DRY_RUN_STEPS
        seeds = [42]
        print(
            "[EXQ-224] DRY-RUN mode:"
            f" episodes={n_episodes} steps={steps_per_ep} seeds={seeds}",
            flush=True,
        )

    print(
        f"\n[EXQ-224] MECH-124 z_goal salience diagnostic"
        f"  episodes={n_episodes}  steps/ep={steps_per_ep}"
        f"  seeds={seeds}",
        flush=True,
    )
    print(
        f"[EXQ-224] Checkpoints every {CHECKPOINT_EVERY} episodes",
        flush=True,
    )

    all_baseline: List[Dict] = []
    all_ablation: List[Dict] = []

    for seed in seeds:
        print(f"\n[EXQ-224] seed={seed}  Condition: BASELINE (z_goal_enabled=True)", flush=True)
        baseline = _run_condition(
            label=f"BASELINE_s{seed}",
            seed=seed,
            n_episodes=n_episodes,
            steps_per_ep=steps_per_ep,
            z_goal_enabled=True,
        )
        all_baseline.append(baseline)

        print(f"\n[EXQ-224] seed={seed}  Condition: ABLATION (z_goal_enabled=False)", flush=True)
        ablation = _run_condition(
            label=f"ABLATION_s{seed}",
            seed=seed,
            n_episodes=n_episodes,
            steps_per_ep=steps_per_ep,
            z_goal_enabled=False,
        )
        all_ablation.append(ablation)

    # --- Aggregate across seeds ---
    def _mean_field(results: List[Dict], field: str) -> float:
        vals = [r[field] for r in results if field in r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    mean_ratio_final  = _mean_field(all_baseline, "final_ratio")
    mean_ratio_slope  = _mean_field(all_baseline, "ratio_slope")
    mean_goal_act     = _mean_field(all_baseline, "final_goal_active_frac")
    mean_res_rate     = _mean_field(all_baseline, "final_resource_rate")
    mean_harm_rate    = _mean_field(all_baseline, "final_harm_rate")

    # --- Diagnostic flags ---
    flag_1 = mean_ratio_final < 0.2       # harm dominates z_goal
    flag_2 = mean_ratio_slope < -0.002    # z_goal salience eroding
    flag_3 = mean_goal_act < 0.3          # goal state rarely active
    risk_detected = flag_1 or flag_2

    # --- Status (diagnostic: PASS=no risk, FAIL=risk detected) ---
    status = "FAIL" if risk_detected else "PASS"

    print(f"\n[EXQ-224] -- Diagnostic Results --", flush=True)
    print(f"  mean_ratio_final:  {mean_ratio_final:.3f}  (FLAG_1 < 0.2: {flag_1})", flush=True)
    print(f"  mean_ratio_slope:  {mean_ratio_slope:.5f} /ep  (FLAG_2 < -0.002: {flag_2})", flush=True)
    print(f"  mean_goal_active:  {mean_goal_act:.3f}  (FLAG_3 < 0.3: {flag_3})", flush=True)
    print(f"  mean_resource_rate: {mean_res_rate:.4f}", flush=True)
    print(f"  mean_harm_rate:     {mean_harm_rate:.4f}", flush=True)
    print(f"  MECH-124 V4 risk_detected: {risk_detected}", flush=True)
    print(f"  Status: {status}", flush=True)

    if risk_detected:
        print("\n[EXQ-224] WARNING: V4 risk pattern detected.", flush=True)
        print(
            "  z_goal is not maintaining competitive salience with harm.",
            flush=True,
        )
        print(
            "  V4 consolidation (MECH-121) will amplify this imbalance.",
            flush=True,
        )
        print(
            "  Recommended: add balanced replay scheduling to MECH-121 design.",
            flush=True,
        )
    else:
        print("\n[EXQ-224] No V4 risk detected: z_goal salience competitive.", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    result = {
        "status":             status,
        "run_id":             run_id,
        "experiment_type":    EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "metrics": {
            "seeds":              seeds,
            "n_episodes":         n_episodes,
            "steps_per_ep":       steps_per_ep,
            "mean_ratio_final":   mean_ratio_final,
            "mean_ratio_slope":   mean_ratio_slope,
            "mean_goal_active_frac": mean_goal_act,
            "mean_resource_rate": mean_res_rate,
            "mean_harm_rate":     mean_harm_rate,
            "baseline_results":   all_baseline,
            "ablation_results":   all_ablation,
        },
        "diagnostic_flags": {
            "FLAG_1_harm_dominated":  flag_1,
            "FLAG_2_ratio_declining": flag_2,
            "FLAG_3_goal_inactive":   flag_3,
            "risk_detected":          risk_detected,
        },
        "thresholds": {
            "FLAG_1_threshold": 0.2,
            "FLAG_2_threshold": -0.002,
            "FLAG_3_threshold": 0.3,
        },
        "interpretation": (
            "MECH-124 V4 risk detected -- add balanced replay scheduling (MECH-121)"
            if risk_detected
            else "No MECH-124 V4 risk -- z_goal salience competitive with harm"
        ),
        "dry_run": dry_run,
    }

    if dry_run:
        print("[EXQ-224] Dry-run complete. No output written.", flush=True)
        return result

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {status}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EXQ-224 MECH-124 z_goal salience diagnostic"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Short smoke test (4 episodes x 10 steps, 1 seed, no output written)"
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help=f"Number of training episodes (default {DEFAULT_EPISODES})"
    )
    parser.add_argument(
        "--steps", type=int, default=STEPS_PER_EP,
        help=f"Steps per episode (default {STEPS_PER_EP})"
    )
    args = parser.parse_args()

    run(
        n_episodes=args.episodes,
        steps_per_ep=args.steps,
        dry_run=args.dry_run,
    )
