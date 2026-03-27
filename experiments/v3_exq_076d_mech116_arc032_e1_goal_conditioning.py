#!/opt/local/bin/python3
"""
V3-EXQ-076d -- MECH-116 / ARC-032: E1 Goal Conditioning (navigation-seeded fix)

Claims: MECH-116, ARC-032
Supersedes: V3-EXQ-076c

Root cause of 076b/076c ERRORs and FAILs:
  076b/076c crashed with exit code 1 at startup (historical backward() inplace-op bug,
  fixed 2026-03-24). When re-run after the fix, goal_norm=0 throughout both conditions
  because the random-walk policy never reaches resources in 2000 steps on an 8x8 grid,
  so benefit_exp_val stays 0 and z_goal never seeds. Goal persistence (C1/C3/C4)
  cannot be tested if z_goal is never activated.

This version fixes both issues:
  (1) ID rename prevents stuck-ID skip in runner_status.json.
  (2) CausalGridWorldV2 size=6 (smaller grid, resource_respawn_on_consume=True):
      resources reappear after collection.
  (3) 50% proximity-following action selection: agent moves greedily toward nearest
      resource 50% of steps, ensuring resources are visited and z_goal is seeded
      before the removal test at step 1000.

MECH-116 asserts that E1's LSTM hidden state, when conditioned on z_goal_latent,
maintains goal context recurrently across steps -- even after benefit signals stop.

ARC-032 asserts that the theta-rate ThetaBuffer (MECH-089) is the primary pathway
through which E1's goal context reaches E3's trajectory scoring.

Key test: resource removed at step 1000. With E1 goal conditioning, z_goal should
persist for more steps (longer half-life) than without conditioning.

Two conditions (matched seeds):
  A. wanting_noe1  -- z_goal_enabled=True, e1_goal_conditioned=False
  B. wanting_withe1 -- z_goal_enabled=True, e1_goal_conditioned=True

PASS criteria (ALL required, identical to 076c):
  C1: goal_halflife_withe1 >= 1.5 * goal_halflife_noe1
  C2: resource_rate_withe1 >= resource_rate_noe1 + 0.03
  C3: goal_norm_noe1_t1200 < 0.5 * goal_norm_noe1_peak
  C4: goal_norm_withe1_t1200 >= goal_norm_noe1_t1200 + 0.02
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


EXPERIMENT_TYPE = "v3_exq_076d_mech116_arc032_e1_goal_conditioning"
CLAIM_IDS = ["MECH-116", "ARC-032"]

GRID_SIZE = 6
TOTAL_STEPS = 2000
REMOVAL_STEP = 1000
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

    e1_goal_conditioned = condition == "wanting_withe1"
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
    goal_norm_at_1200: float = 0.0

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
        # Resource removal at step 1000
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

        # 50% greedy toward nearest resource, 50% random
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

        # E1 prediction error tracking
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

        if step == 1200:
            diag = agent.compute_goal_maintenance_diagnostic()
            goal_norm_at_1200 = diag["goal_norm"]

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

        if step % 500 == 0:
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

    goal_halflife = TOTAL_STEPS
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
        f" goal_norm_t1200={goal_norm_at_1200:.4f}",
        flush=True,
    )

    return {
        "condition": condition,
        "resource_visit_rate": resource_visit_rate,
        "harm_rate": harm_rate,
        "goal_halflife": goal_halflife,
        "peak_goal_norm": peak_goal_norm,
        "peak_goal_prox": peak_prox,
        "goal_norm_at_1200": goal_norm_at_1200,
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
    """MECH-116 / ARC-032: E1 goal conditioning -- noe1 vs withe1."""
    print(
        f"\n[EXQ-076d] MECH-116/ARC-032 E1 Goal Conditioning seed={seed}",
        flush=True,
    )

    r_noe1   = _run_condition("wanting_noe1",   seed, world_dim, alpha_world, lr)
    r_withe1 = _run_condition("wanting_withe1", seed, world_dim, alpha_world, lr)

    # PASS criteria (identical to 076c)
    c1_pass = r_withe1["goal_halflife"] >= 1.5 * r_noe1["goal_halflife"]
    c2_pass = r_withe1["resource_visit_rate"] >= r_noe1["resource_visit_rate"] + 0.03
    c3_pass = (
        r_noe1["goal_norm_at_1200"] < 0.5 * r_noe1["peak_goal_norm"]
        if r_noe1["peak_goal_norm"] > 1e-6
        else False
    )
    c4_pass = r_withe1["goal_norm_at_1200"] >= r_noe1["goal_norm_at_1200"] + 0.02

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[EXQ-076d] Results:", flush=True)
    print(
        f"  goal_halflife: noe1={r_noe1['goal_halflife']}"
        f"  withe1={r_withe1['goal_halflife']}"
        f"  ratio={r_withe1['goal_halflife']/max(1,r_noe1['goal_halflife']):.2f}",
        flush=True,
    )
    print(
        f"  resource_rate: noe1={r_noe1['resource_visit_rate']:.4f}"
        f"  withe1={r_withe1['resource_visit_rate']:.4f}",
        flush=True,
    )
    print(
        f"  goal_norm_t1200: noe1={r_noe1['goal_norm_at_1200']:.4f}"
        f"  withe1={r_withe1['goal_norm_at_1200']:.4f}"
        f"  peak_noe1={r_noe1['peak_goal_norm']:.4f}",
        flush=True,
    )
    print(f"  Status: {status} ({criteria_met}/4)", flush=True)

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
            " from E1 conditioning but below full threshold. May need longer"
            " pre-removal training or larger goal_weight."
        )
    else:
        interpretation = (
            "MECH-116 / ARC-032 NOT SUPPORTED: E1 conditioning does not"
            " demonstrably extend goal persistence. z_goal decay_goal may"
            " dominate over E1 recurrent maintenance, or benefit signal is"
            " insufficient to establish a strong z_goal prior to removal."
        )

    failure_notes: List[str] = []
    if not c1_pass:
        ratio = r_withe1["goal_halflife"] / max(1, r_noe1["goal_halflife"])
        failure_notes.append(
            f"C1 FAIL: halflife ratio={ratio:.2f} < 1.5"
            f" (noe1={r_noe1['goal_halflife']} withe1={r_withe1['goal_halflife']})"
        )
    if not c2_pass:
        gap = r_withe1["resource_visit_rate"] - r_noe1["resource_visit_rate"]
        failure_notes.append(f"C2 FAIL: resource_rate gap={gap:.4f} < 0.03")
    if not c3_pass:
        if r_noe1["peak_goal_norm"] > 1e-6:
            ratio2 = r_noe1["goal_norm_at_1200"] / r_noe1["peak_goal_norm"]
            failure_notes.append(
                f"C3 FAIL: noe1 t1200/peak={ratio2:.3f} >= 0.5 (expected decay)"
            )
        else:
            failure_notes.append(
                "C3 FAIL: peak_goal_norm near zero (goal never activated)"
            )
    if not c4_pass:
        diff = r_withe1["goal_norm_at_1200"] - r_noe1["goal_norm_at_1200"]
        failure_notes.append(f"C4 FAIL: goal_norm_t1200 diff={diff:.4f} < 0.02")
    for n in failure_notes:
        print(f"  {n}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-076d -- MECH-116 / ARC-032 E1 Goal Conditioning\n\n"
        f"**Status:** {status}\n**Claims:** MECH-116, ARC-032\n"
        f"**Seed:** {seed}  **Steps:** {TOTAL_STEPS}"
        f"  **Resource removal:** step {REMOVAL_STEP}\n"
        f"**Fix from 076c:** CausalGridWorldV2 size=6, resource_respawn,"
        f" 50% proximity-following navigation (z_goal seeding fix).\n\n"
        f"## Conditions\n\n"
        f"- wanting_noe1: z_goal_enabled=True, e1_goal_conditioned=False\n"
        f"- wanting_withe1: z_goal_enabled=True, e1_goal_conditioned=True\n\n"
        f"## Results\n\n"
        f"| Metric | noe1 | withe1 |\n|---|---|---|\n"
        f"| resource_visit_rate | {r_noe1['resource_visit_rate']:.4f}"
        f" | {r_withe1['resource_visit_rate']:.4f} |\n"
        f"| goal_halflife (steps) | {r_noe1['goal_halflife']}"
        f" | {r_withe1['goal_halflife']} |\n"
        f"| peak_goal_norm | {r_noe1['peak_goal_norm']:.4f}"
        f" | {r_withe1['peak_goal_norm']:.4f} |\n"
        f"| goal_norm_at_1200 | {r_noe1['goal_norm_at_1200']:.4f}"
        f" | {r_withe1['goal_norm_at_1200']:.4f} |\n"
        f"| harm_rate | {r_noe1['harm_rate']:.4f}"
        f" | {r_withe1['harm_rate']:.4f} |\n\n"
        f"## PASS Criteria\n\n| Criterion | Result |\n|---|---|\n"
        f"| C1: halflife ratio >= 1.5x | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: resource gap >= 0.03 | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| C3: noe1 goal decays to <50% peak by t1200 | {'PASS' if c3_pass else 'FAIL'} |\n"
        f"| C4: withe1 goal_norm_t1200 >= noe1 + 0.02 | {'PASS' if c4_pass else 'FAIL'} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n{interpretation}\n{failure_section}\n"
    )

    metrics = {
        "resource_rate_noe1":          float(r_noe1["resource_visit_rate"]),
        "resource_rate_withe1":        float(r_withe1["resource_visit_rate"]),
        "goal_halflife_noe1":          float(r_noe1["goal_halflife"]),
        "goal_halflife_withe1":        float(r_withe1["goal_halflife"]),
        "halflife_ratio":              float(
            r_withe1["goal_halflife"] / max(1, r_noe1["goal_halflife"])
        ),
        "peak_goal_norm_noe1":         float(r_noe1["peak_goal_norm"]),
        "peak_goal_norm_withe1":       float(r_withe1["peak_goal_norm"]),
        "goal_norm_at_1200_noe1":      float(r_noe1["goal_norm_at_1200"]),
        "goal_norm_at_1200_withe1":    float(r_withe1["goal_norm_at_1200"]),
        "harm_rate_noe1":              float(r_noe1["harm_rate"]),
        "harm_rate_withe1":            float(r_withe1["harm_rate"]),
        "e1_err_goal_noe1":            float(r_noe1["mean_e1_err_goal_steps"]),
        "e1_err_neutral_noe1":         float(r_noe1["mean_e1_err_neutral_steps"]),
        "e1_err_goal_withe1":          float(r_withe1["mean_e1_err_goal_steps"]),
        "e1_err_neutral_withe1":       float(r_withe1["mean_e1_err_neutral_steps"]),
        "crit1_pass":                  1.0 if c1_pass else 0.0,
        "crit2_pass":                  1.0 if c2_pass else 0.0,
        "crit3_pass":                  1.0 if c3_pass else 0.0,
        "crit4_pass":                  1.0 if c4_pass else 0.0,
        "criteria_met":                float(criteria_met),
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
            "wanting_noe1": r_noe1,
            "wanting_withe1": r_withe1,
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
