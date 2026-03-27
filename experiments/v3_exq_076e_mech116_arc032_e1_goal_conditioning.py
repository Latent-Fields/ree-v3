#!/opt/local/bin/python3
"""
V3-EXQ-076e -- MECH-116 / ARC-032: E1 Goal Conditioning (10k-step budget)

Claims: MECH-116, ARC-032
Supersedes: V3-EXQ-076d

Root cause of 076d FAIL:
  076d ran 2000 total steps with resource removal at step 1000 and a 1000-step
  post-removal window. The 30%-peak halflife threshold was never reached in either
  condition -- goal persisted robustly in both (goal_norm decayed from ~0.36 to
  ~0.13 but never hit 30% threshold). Only C3 passed (some decay observable).
  The null result is uninformative: 1000 post-removal steps is too short to
  observe differential decay between GOAL_CONDITIONED and GOAL_UNCONDITIONED.

This version fixes the budget:
  - Total steps: 10,000 (goal well-established by step 3000)
  - Resource removal at step 5000 (leaves 5000-step post-removal window)
  - At 10,000 steps the 30% halflife threshold should be reachable if E1
    conditioning genuinely extends goal persistence

MECH-116 asserts that E1's LSTM hidden state, when conditioned on z_goal_latent,
maintains goal context recurrently across steps -- even after benefit signals stop.

ARC-032 asserts that the theta-rate ThetaBuffer (MECH-089) is the primary pathway
through which E1's goal context reaches E3's trajectory scoring.

Two conditions (matched seeds):
  A. goal_unconditioned -- z_goal_enabled=True, e1_goal_conditioned=False
  B. goal_conditioned   -- z_goal_enabled=True, e1_goal_conditioned=True

PASS criteria (ALL required):
  C1: goal_halflife_conditioned >= 1.5 * goal_halflife_unconditioned
  C2: resource_rate_conditioned >= resource_rate_unconditioned + 0.03
  C3: goal_norm_conditioned > 0.05 at step REMOVAL_STEP + 1000 (= 6000)
"""

import sys
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


EXPERIMENT_TYPE = "v3_exq_076e_mech116_arc032_e1_goal_conditioning"
CLAIM_IDS = ["MECH-116", "ARC-032"]

GRID_SIZE = 6
TOTAL_STEPS = 10000
REMOVAL_STEP = 5000
GOAL_HALFLIFE_THRESHOLD = 0.3
GREEDY_FRAC = 0.5


def _greedy_action_toward_resource(env) -> int:
    """Return action index that reduces L1 distance to nearest resource.
    CausalGridWorldV2 actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1),
                               3=right(0,+1), 4=stay(0,0).
    Falls back to random if no resources or already at resource.
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
        size=GRID_SIZE,
        num_resources=2,
        num_hazards=2,
        hazard_harm=0.02,
        resource_benefit=0.3,
        resource_respawn_on_consume=True,
    )


def _run_condition(
    condition: str,
    seed: int,
    world_dim: int = 32,
    alpha_world: float = 0.9,
    lr: float = 1e-3,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)

    e1_goal_conditioned = condition == "goal_conditioned"
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        z_goal_enabled=True,
        e1_goal_conditioned=e1_goal_conditioned,
        goal_weight=1.0,
    )
    agent = REEAgent(config)
    device = agent.device

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )

    agent.train()

    resource_removed = False

    resource_visits = 0
    harm_events = 0
    total_steps = 0
    pre_removal_steps = 0

    goal_norm_series: List[float] = []
    goal_prox_series: List[float] = []
    goal_norm_times: List[int] = []

    peak_goal_norm: float = 0.0
    goal_norm_at_r1000: float = 0.0  # goal_norm at REMOVAL_STEP + 1000

    e1_err_goal_steps: List[float] = []
    e1_err_neutral_steps: List[float] = []

    resource_loc: Optional[tuple] = None
    if env.resources:
        resource_loc = (int(env.resources[0][0]), int(env.resources[0][1]))

    _, obs_dict = env.reset()
    agent.reset()

    if env.resources:
        resource_loc = (int(env.resources[0][0]), int(env.resources[0][1]))

    step = 0
    while step < TOTAL_STEPS:
        # Resource removal at REMOVAL_STEP
        if step == REMOVAL_STEP and not resource_removed:
            for rx, ry in env.resources:
                env.grid[rx, ry] = env.ENTITY_TYPES["empty"]
            env.resources = []
            env._compute_proximity_fields()
            resource_removed = True
            goal_now = agent.goal_state.goal_norm() if agent.goal_state else 0.0
            print(
                f"  [{condition}] step={step}: resources removed."
                f" goal_norm={goal_now:.4f}",
                flush=True,
            )

        obs_body = obs_dict["body_state"].to(device)
        obs_world = obs_dict["world_state"].to(device)

        z_self_prev = None
        if agent._current_latent is not None:
            z_self_prev = agent._current_latent.z_self.detach().clone()

        latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()

        # 50% greedy toward nearest resource (ensures z_goal seeding pre-removal),
        # 50% random
        if random.random() < GREEDY_FRAC and not resource_removed:
            action_idx = _greedy_action_toward_resource(env)
        else:
            action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)

        if z_self_prev is not None:
            agent.record_transition(z_self_prev, action, latent.z_self.detach())

        _, reward, done, info, obs_dict = env.step(action)
        harm_signal = float(reward) if float(reward) < 0 else 0.0
        ttype = info.get("transition_type", "none")

        if ttype == "resource":
            resource_visits += 1
        if ttype in ("agent_caused_hazard", "hazard_approach"):
            harm_events += 1
        if step < REMOVAL_STEP:
            pre_removal_steps += 1

        # E1 prediction error tracking (near goal vs neutral)
        if resource_loc is not None and agent._current_latent is not None:
            ax2, ay2 = env.agent_x, env.agent_y
            d_to_res = abs(ax2 - resource_loc[0]) + abs(ay2 - resource_loc[1])
            e1_loss_val = agent.compute_prediction_loss()
            e1_err = float(e1_loss_val.item()) if e1_loss_val.requires_grad else 0.0
            if d_to_res <= 3:
                e1_err_goal_steps.append(e1_err)
            else:
                e1_err_neutral_steps.append(e1_err)

        # E1 + E2 combined backward
        e1_loss_train = agent.compute_prediction_loss()
        e2_loss = agent.compute_e2_loss()
        total_e1_e2 = e1_loss_train + e2_loss
        if total_e1_e2.requires_grad:
            e1_opt.zero_grad()
            e2_opt.zero_grad()
            total_e1_e2.backward()
            e1_opt.step()
            e2_opt.step()

        # E3 harm supervision
        if agent._current_latent is not None:
            z_world = agent._current_latent.z_world.detach()
            harm_target = torch.tensor(
                [[1.0 if harm_signal < 0 else 0.0]], device=device
            )
            harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
            e3_opt.zero_grad()
            harm_loss.backward()
            e3_opt.step()

        # Wanting: z_goal update from benefit signal
        benefit_exp_val = float(obs_body[11])
        agent.update_z_goal(benefit_exp_val)
        agent.update_residue(harm_signal)

        # Track goal state every 10 steps
        if step % 10 == 0:
            diag = agent.compute_goal_maintenance_diagnostic()
            goal_norm_series.append(diag["goal_norm"])
            goal_prox_series.append(diag["goal_proximity"])
            goal_norm_times.append(step)
            if diag["goal_norm"] > peak_goal_norm:
                peak_goal_norm = diag["goal_norm"]

        # Checkpoint: REMOVAL_STEP + 1000
        if step == REMOVAL_STEP + 1000:
            diag = agent.compute_goal_maintenance_diagnostic()
            goal_norm_at_r1000 = diag["goal_norm"]

        total_steps += 1
        step += 1

        if done:
            _, obs_dict = env.reset()
            agent.reset()
            if resource_removed:
                for rx2, ry2 in env.resources:
                    env.grid[rx2, ry2] = env.ENTITY_TYPES["empty"]
                env.resources = []
                env._compute_proximity_fields()
            elif env.resources:
                resource_loc = (int(env.resources[0][0]), int(env.resources[0][1]))

        if step % 1000 == 0:
            diag = agent.compute_goal_maintenance_diagnostic()
            print(
                f"  [{condition}] step={step}/{TOTAL_STEPS}"
                f" resources={resource_visits} harm={harm_events}"
                f" goal_norm={diag['goal_norm']:.4f}"
                f" goal_prox={diag['goal_proximity']:.4f}",
                flush=True,
            )

    resource_visit_rate = resource_visits / max(1, pre_removal_steps)
    harm_rate = harm_events / max(1, total_steps)

    # Goal half-life: steps after removal until goal_prox < threshold * peak_prox
    pre_removal_prox = [
        goal_prox_series[i] for i, t in enumerate(goal_norm_times) if t < REMOVAL_STEP
    ]
    peak_prox = max(pre_removal_prox) if pre_removal_prox else 1e-6
    halflife_threshold = GOAL_HALFLIFE_THRESHOLD * peak_prox

    goal_halflife = TOTAL_STEPS  # sentinel: threshold never reached
    for i, t in enumerate(goal_norm_times):
        if t >= REMOVAL_STEP and goal_prox_series[i] < halflife_threshold:
            goal_halflife = t - REMOVAL_STEP
            break

    mean_e1_err_goal = sum(e1_err_goal_steps) / max(1, len(e1_err_goal_steps))
    mean_e1_err_neutral = sum(e1_err_neutral_steps) / max(1, len(e1_err_neutral_steps))

    print(
        f"  [{condition}] DONE: resource_rate={resource_visit_rate:.4f}"
        f" harm_rate={harm_rate:.4f}"
        f" goal_halflife={goal_halflife}"
        f" peak_prox={peak_prox:.4f}"
        f" goal_norm_at_r1000={goal_norm_at_r1000:.4f}",
        flush=True,
    )

    return {
        "condition": condition,
        "resource_visit_rate": resource_visit_rate,
        "harm_rate": harm_rate,
        "goal_halflife": goal_halflife,
        "peak_goal_norm": peak_goal_norm,
        "peak_goal_prox": peak_prox,
        "goal_norm_at_r1000": goal_norm_at_r1000,
        "mean_e1_err_goal_steps": mean_e1_err_goal,
        "mean_e1_err_neutral_steps": mean_e1_err_neutral,
        "n_e1_goal_steps": len(e1_err_goal_steps),
        "n_e1_neutral_steps": len(e1_err_neutral_steps),
        "n_resource_visits": resource_visits,
        "n_harm_events": harm_events,
        "total_steps": total_steps,
    }


def run(
    seed: int = 42,
    world_dim: int = 32,
    alpha_world: float = 0.9,
    lr: float = 1e-3,
    **kwargs,
) -> dict:
    """MECH-116 / ARC-032: E1 goal conditioning -- unconditioned vs conditioned."""
    print(
        f"\n[EXQ-076e] MECH-116/ARC-032 E1 Goal Conditioning 10k-step budget"
        f" seed={seed}",
        flush=True,
    )

    r_uncond = _run_condition("goal_unconditioned", seed, world_dim, alpha_world, lr)
    r_cond   = _run_condition("goal_conditioned",   seed, world_dim, alpha_world, lr)

    # PASS criteria
    c1_pass = r_cond["goal_halflife"] >= 1.5 * r_uncond["goal_halflife"]
    c2_pass = r_cond["resource_visit_rate"] >= r_uncond["resource_visit_rate"] + 0.03
    c3_pass = r_cond["goal_norm_at_r1000"] > 0.05

    all_pass = c1_pass and c2_pass and c3_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[EXQ-076e] Results:", flush=True)
    print(
        f"  goal_halflife: uncond={r_uncond['goal_halflife']}"
        f"  cond={r_cond['goal_halflife']}"
        f"  ratio={r_cond['goal_halflife']/max(1,r_uncond['goal_halflife']):.2f}",
        flush=True,
    )
    print(
        f"  resource_rate: uncond={r_uncond['resource_visit_rate']:.4f}"
        f"  cond={r_cond['resource_visit_rate']:.4f}",
        flush=True,
    )
    print(
        f"  goal_norm_at_r1000: uncond={r_uncond['goal_norm_at_r1000']:.4f}"
        f"  cond={r_cond['goal_norm_at_r1000']:.4f}",
        flush=True,
    )
    print(f"  Status: {status} ({criteria_met}/3)", flush=True)

    if all_pass:
        interpretation = (
            "MECH-116 / ARC-032 SUPPORTED: E1 LSTM goal conditioning extends"
            " z_goal persistence after resource removal. The theta-rate ThetaBuffer"
            " pathway carries goal context from E1 to E3, counteracting purely"
            " passive decay. Consistent with DLPFC sustained firing (Fuster &"
            " Alexander 1971) and frontal-hippocampal theta coupling (Hyman 2010)."
        )
    elif criteria_met >= 2:
        interpretation = (
            "MECH-116 / ARC-032 PARTIAL: Some goal maintenance advantage"
            " from E1 conditioning but below full threshold. May need further"
            " budget increase or stronger goal_weight."
        )
    else:
        interpretation = (
            "MECH-116 / ARC-032 NOT SUPPORTED: E1 conditioning does not"
            " demonstrably extend goal persistence at 10,000-step budget."
            " z_goal decay may dominate over E1 recurrent maintenance, or"
            " benefit signal is insufficient to establish a strong z_goal"
            " prior to removal."
        )

    failure_notes: List[str] = []
    if not c1_pass:
        ratio = r_cond["goal_halflife"] / max(1, r_uncond["goal_halflife"])
        failure_notes.append(
            f"C1 FAIL: halflife ratio={ratio:.2f} < 1.5"
            f" (uncond={r_uncond['goal_halflife']} cond={r_cond['goal_halflife']})"
        )
    if not c2_pass:
        gap = r_cond["resource_visit_rate"] - r_uncond["resource_visit_rate"]
        failure_notes.append(f"C2 FAIL: resource_rate gap={gap:.4f} < 0.03")
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: goal_norm_conditioned_at_r1000={r_cond['goal_norm_at_r1000']:.4f}"
            f" <= 0.05 (E1 conditioning not maintaining detectable goal signal"
            f" at step {REMOVAL_STEP + 1000})"
        )
    for n in failure_notes:
        print(f"  {n}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-076e -- MECH-116 / ARC-032 E1 Goal Conditioning 10k-step\n\n"
        f"**Status:** {status}\n**Claims:** MECH-116, ARC-032\n"
        f"**Seed:** {seed}  **Steps:** {TOTAL_STEPS}"
        f"  **Resource removal:** step {REMOVAL_STEP}"
        f"  **Checkpoint:** step {REMOVAL_STEP + 1000}\n"
        f"**Budget rationale:** 076d used 2000 steps (removal at 1000); 30%% halflife"
        f" threshold never reached in either condition. Extended to 10,000 steps"
        f" (removal at 5000) to allow differential decay to emerge.\n\n"
        f"## Conditions\n\n"
        f"- goal_unconditioned: z_goal_enabled=True, e1_goal_conditioned=False\n"
        f"- goal_conditioned: z_goal_enabled=True, e1_goal_conditioned=True\n\n"
        f"## Results\n\n"
        f"| Metric | unconditioned | conditioned |\n|---|---|---|\n"
        f"| resource_visit_rate (pre-removal) | {r_uncond['resource_visit_rate']:.4f}"
        f" | {r_cond['resource_visit_rate']:.4f} |\n"
        f"| goal_halflife (steps post-removal) | {r_uncond['goal_halflife']}"
        f" | {r_cond['goal_halflife']} |\n"
        f"| halflife ratio | --"
        f" | {r_cond['goal_halflife']/max(1,r_uncond['goal_halflife']):.2f} |\n"
        f"| peak_goal_norm | {r_uncond['peak_goal_norm']:.4f}"
        f" | {r_cond['peak_goal_norm']:.4f} |\n"
        f"| goal_norm at step {REMOVAL_STEP + 1000} | {r_uncond['goal_norm_at_r1000']:.4f}"
        f" | {r_cond['goal_norm_at_r1000']:.4f} |\n"
        f"| harm_rate | {r_uncond['harm_rate']:.4f}"
        f" | {r_cond['harm_rate']:.4f} |\n\n"
        f"## PASS Criteria\n\n| Criterion | Result |\n|---|---|\n"
        f"| C1: halflife ratio >= 1.5x | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: resource gap >= 0.03 | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| C3: goal_norm_conditioned > 0.05 at step {REMOVAL_STEP + 1000}"
        f" | {'PASS' if c3_pass else 'FAIL'} |\n\n"
        f"Criteria met: {criteria_met}/3 -> **{status}**\n\n"
        f"## Interpretation\n\n{interpretation}\n{failure_section}\n"
    )

    metrics = {
        "resource_rate_unconditioned":     float(r_uncond["resource_visit_rate"]),
        "resource_rate_conditioned":       float(r_cond["resource_visit_rate"]),
        "goal_halflife_unconditioned":     float(r_uncond["goal_halflife"]),
        "goal_halflife_conditioned":       float(r_cond["goal_halflife"]),
        "halflife_ratio":                  float(
            r_cond["goal_halflife"] / max(1, r_uncond["goal_halflife"])
        ),
        "peak_goal_norm_unconditioned":    float(r_uncond["peak_goal_norm"]),
        "peak_goal_norm_conditioned":      float(r_cond["peak_goal_norm"]),
        "goal_norm_at_r1000_unconditioned": float(r_uncond["goal_norm_at_r1000"]),
        "goal_norm_at_r1000_conditioned":  float(r_cond["goal_norm_at_r1000"]),
        "harm_rate_unconditioned":         float(r_uncond["harm_rate"]),
        "harm_rate_conditioned":           float(r_cond["harm_rate"]),
        "e1_err_goal_unconditioned":       float(r_uncond["mean_e1_err_goal_steps"]),
        "e1_err_neutral_unconditioned":    float(r_uncond["mean_e1_err_neutral_steps"]),
        "e1_err_goal_conditioned":         float(r_cond["mean_e1_err_goal_steps"]),
        "e1_err_neutral_conditioned":      float(r_cond["mean_e1_err_neutral_steps"]),
        "crit1_pass":                      1.0 if c1_pass else 0.0,
        "crit2_pass":                      1.0 if c2_pass else 0.0,
        "crit3_pass":                      1.0 if c3_pass else 0.0,
        "criteria_met":                    float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
        "condition_results": {
            "goal_unconditioned": r_uncond,
            "goal_conditioned": r_cond,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--world-dim",   type=int,   default=32)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--lr",          type=float, default=1e-3)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        world_dim=args.world_dim,
        alpha_world=args.alpha_world,
        lr=args.lr,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
