#!/opt/local/bin/python3
"""
V3-EXQ-074e -- MECH-117 Wanting/Liking Dissociation (navigation-seeded fix)

Claims: MECH-112, MECH-117
Supersedes: V3-EXQ-074d

Root cause of 074b/074c ERRORs and FAILs:
  074b/074c crashed with exit code 1 at startup (historical backward() inplace-op bug,
  fixed 2026-03-24). When re-run after the fix, resource_rate=0 for ALL conditions
  including nogo -- pure random walk on 8x8 grid never reaches resources in 3000 steps,
  so benefit_exp_val=0 throughout and z_goal never seeds. The wanting/liking dissociation
  cannot manifest without a seeded z_goal.

This version fixes both issues:
  (1) ID rename prevents stuck-ID skip in runner_status.json.
  (2) CausalGridWorldV2 size=6 (smaller grid, resource_respawn_on_consume=True):
      resources reappear after collection, so even random walk finds them eventually.
  (3) 50% proximity-following action selection: agent moves greedily toward nearest
      resource 50% of steps, ensuring resources are visited and z_goal seeds before
      the dissociation test begins.

Tests the functional dissociation between:
  - Liking (benefit_eval_head): learned from benefit_exposure at receipt;
    sharp proximity spike; redirects rapidly when resource moves.
  - Wanting (z_goal_latent): slow-decay attractor; continuous spatial
    gradient; persists after resource relocation (incentive salience).

Three conditions (matched seeds):
  A. nogo    -- benefit_eval_enabled=False, z_goal_enabled=False (baseline)
  B. liking  -- benefit_eval_enabled=True,  z_goal_enabled=False
  C. wanting -- benefit_eval_enabled=False, z_goal_enabled=True, e1_goal_conditioned=True

Each condition runs for 3000 steps on a 6x6 grid.
At step 1500: primary resource cell (L1) is relocated to L2 (3+ cells away).

PASS criteria (ALL required, identical to 074c):
  C1: wanting_resource_rate >= nogo_resource_rate + 0.04
  C2: liking_l2_redirect <= 15 steps
  C3: wanting_l1_fraction >= 0.25 (25% of steps 1500-1700 directed toward L1)
  C4: wanting_harm_rate <= nogo_harm_rate + 0.03
"""

import sys
import random
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_074e_mech112_117_wanting_liking"
CLAIM_IDS = ["MECH-112", "MECH-117"]

GRID_SIZE = 6
TOTAL_STEPS = 3000
RELOCATION_STEP = 1500
# Fraction of steps that use greedy proximity-following (vs random)
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
        return 1 if dx > 0 else 0   # down or up
    else:
        return 3 if dy > 0 else 2   # right or left


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


def _make_agent(
    condition: str,
    env: CausalGridWorldV2,
    world_dim: int = 32,
    alpha_world: float = 0.9,
    lr: float = 1e-3,
) -> REEAgent:
    kwargs: dict = dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
    )
    if condition == "liking":
        kwargs["benefit_eval_enabled"] = True
        kwargs["benefit_weight"] = 1.0
    elif condition == "wanting":
        kwargs["z_goal_enabled"] = True
        kwargs["e1_goal_conditioned"] = True
        kwargs["goal_weight"] = 1.0

    config = REEConfig.from_dims(**kwargs)
    return REEAgent(config)


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
    agent = _make_agent(condition, env, world_dim, alpha_world, lr)
    device = agent.device

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )

    agent.train()

    # Track L1 (original resource cell) and L2 (new after relocation)
    l1_pos: Optional[Tuple[int, int]] = None
    l2_pos: Optional[Tuple[int, int]] = None
    resource_relocated = False

    # Metrics
    resource_visits = 0
    harm_events = 0
    total_steps = 0
    post_reloc_steps = 0
    l1_directed_steps = 0
    liking_l2_redirect_found = False
    liking_l2_redirect_at: Optional[int] = None

    _, obs_dict = env.reset()
    agent.reset()

    if env.resources:
        l1_pos = (int(env.resources[0][0]), int(env.resources[0][1]))

    step = 0
    while step < TOTAL_STEPS:
        # Resource relocation at step 1500
        if step == RELOCATION_STEP and not resource_relocated and l1_pos is not None:
            cx, cy = l1_pos
            new_pos = None
            for dx in range(-(GRID_SIZE - 1), GRID_SIZE):
                for dy in range(-(GRID_SIZE - 1), GRID_SIZE):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        if abs(dx) + abs(dy) >= 3:
                            if env.grid[nx, ny] == env.ENTITY_TYPES["empty"]:
                                new_pos = (nx, ny)
                                break
                if new_pos is not None:
                    break
            if new_pos is None:
                for nx in range(GRID_SIZE):
                    for ny in range(GRID_SIZE):
                        if env.grid[nx, ny] == env.ENTITY_TYPES["empty"]:
                            new_pos = (nx, ny)
                            break
                    if new_pos is not None:
                        break
            if new_pos is not None:
                l2_pos = new_pos
                if env.resources:
                    old_x, old_y = env.resources[0]
                    env.grid[old_x, old_y] = env.ENTITY_TYPES["empty"]
                    env.resources[0] = [l2_pos[0], l2_pos[1]]
                    env.grid[l2_pos[0], l2_pos[1]] = env.ENTITY_TYPES["resource"]
                    env._compute_proximity_fields()
                resource_relocated = True
                print(
                    f"  [{condition}] step={step}:"
                    f" L1=({cx},{cy}) -> L2=({l2_pos[0]},{l2_pos[1]})",
                    flush=True,
                )

        obs_body = obs_dict["body_state"].to(device)
        obs_world = obs_dict["world_state"].to(device)

        # Capture z_self before sense() for E2 transition recording
        z_self_prev = None
        if agent._current_latent is not None:
            z_self_prev = agent._current_latent.z_self.detach().clone()

        latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()

        # 50% greedy toward nearest resource, 50% random
        if random.random() < GREEDY_FRAC:
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

        # Post-relocation L1/L2 tracking
        if resource_relocated and RELOCATION_STEP <= step < RELOCATION_STEP + 200:
            post_reloc_steps += 1
            ax2, ay2 = env.agent_x, env.agent_y
            if l1_pos is not None and l2_pos is not None:
                d_l1 = abs(ax2 - l1_pos[0]) + abs(ay2 - l1_pos[1])
                d_l2 = abs(ax2 - l2_pos[0]) + abs(ay2 - l2_pos[1])
                if d_l1 < d_l2:
                    l1_directed_steps += 1
                elif d_l2 == 0 and not liking_l2_redirect_found:
                    liking_l2_redirect_found = True
                    liking_l2_redirect_at = step - RELOCATION_STEP

        benefit_exp_val = float(obs_body[11])

        # E1 + E2 combined backward (avoids inplace op conflict)
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
            z_world = agent._current_latent.z_world.detach()
            harm_target = torch.tensor(
                [[1.0 if harm_signal < 0 else 0.0]], device=device
            )
            harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
            e3_opt.zero_grad()
            harm_loss.backward()
            e3_opt.step()

        # Liking: benefit_eval supervision
        if condition == "liking" and agent._current_latent is not None:
            benefit_target = torch.tensor([[benefit_exp_val]], device=device)
            benefit_loss = agent.compute_benefit_eval_loss(benefit_target)
            if benefit_loss.requires_grad:
                e3_opt.zero_grad()
                benefit_loss.backward()
                e3_opt.step()

        # Wanting: z_goal update from benefit signal
        if condition == "wanting":
            agent.update_z_goal(benefit_exp_val)

        agent.update_residue(harm_signal)

        total_steps += 1
        step += 1

        if done:
            _, obs_dict = env.reset()
            agent.reset()
            if resource_relocated and l2_pos is not None:
                if env.resources:
                    old_x2, old_y2 = env.resources[0]
                    env.grid[old_x2, old_y2] = env.ENTITY_TYPES["empty"]
                    env.resources[0] = [l2_pos[0], l2_pos[1]]
                    env.grid[l2_pos[0], l2_pos[1]] = env.ENTITY_TYPES["resource"]
                    env._compute_proximity_fields()

        if step % 500 == 0:
            diag = agent.compute_goal_maintenance_diagnostic()
            print(
                f"  [{condition}] step={step}/{TOTAL_STEPS}"
                f" resources={resource_visits} harm={harm_events}"
                f" goal_norm={diag['goal_norm']:.3f}"
                f" goal_prox={diag['goal_proximity']:.3f}",
                flush=True,
            )

    resource_visit_rate = resource_visits / max(1, total_steps)
    harm_rate = harm_events / max(1, total_steps)
    l1_fraction = l1_directed_steps / max(1, post_reloc_steps)
    redirect_steps = liking_l2_redirect_at if liking_l2_redirect_found else 200

    diag = agent.compute_goal_maintenance_diagnostic()

    print(
        f"  [{condition}] DONE: resource_rate={resource_visit_rate:.4f}"
        f" harm_rate={harm_rate:.4f}"
        f" l1_fraction={l1_fraction:.3f}"
        f" l2_redirect={redirect_steps}",
        flush=True,
    )

    return {
        "condition": condition,
        "resource_visit_rate": resource_visit_rate,
        "harm_rate": harm_rate,
        "l1_directed_fraction": l1_fraction,
        "liking_l2_redirect_steps": redirect_steps,
        "goal_norm_final": diag["goal_norm"],
        "goal_proximity_final": diag["goal_proximity"],
        "goal_active": diag["is_active"],
        "n_resource_visits": resource_visits,
        "n_harm_events": harm_events,
        "total_steps": total_steps,
        "post_reloc_steps": post_reloc_steps,
        "l1_directed_steps": l1_directed_steps,
    }


def run(
    seed: int = 42,
    world_dim: int = 32,
    alpha_world: float = 0.9,
    lr: float = 1e-3,
    **kwargs,
) -> dict:
    """MECH-117: wanting/liking dissociation -- three conditions."""
    print(f"\n[EXQ-074e] MECH-117 Wanting/Liking Dissociation seed={seed}", flush=True)

    r_nogo    = _run_condition("nogo",    seed, world_dim, alpha_world, lr)
    r_liking  = _run_condition("liking",  seed, world_dim, alpha_world, lr)
    r_wanting = _run_condition("wanting", seed, world_dim, alpha_world, lr)

    # PASS criteria (identical to 074c)
    c1_pass = r_wanting["resource_visit_rate"] >= r_nogo["resource_visit_rate"] + 0.04
    c2_pass = r_liking["liking_l2_redirect_steps"] <= 15
    c3_pass = r_wanting["l1_directed_fraction"] >= 0.25
    c4_pass = r_wanting["harm_rate"] <= r_nogo["harm_rate"] + 0.03

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[EXQ-074e] Results:", flush=True)
    print(
        f"  resource_rate: nogo={r_nogo['resource_visit_rate']:.4f}"
        f"  liking={r_liking['resource_visit_rate']:.4f}"
        f"  wanting={r_wanting['resource_visit_rate']:.4f}",
        flush=True,
    )
    print(
        f"  l2_redirect (liking): {r_liking['liking_l2_redirect_steps']} steps"
        f"  (threshold: <=15)",
        flush=True,
    )
    print(
        f"  l1_fraction (wanting): {r_wanting['l1_directed_fraction']:.3f}"
        f"  (threshold: >=0.25)",
        flush=True,
    )
    print(
        f"  harm: nogo={r_nogo['harm_rate']:.4f}"
        f"  wanting={r_wanting['harm_rate']:.4f}"
        f"  delta={r_wanting['harm_rate']-r_nogo['harm_rate']:+.4f}",
        flush=True,
    )
    print(f"  Status: {status} ({criteria_met}/4)", flush=True)

    if all_pass:
        interpretation = (
            "MECH-117 SUPPORTED: wanting (z_goal_latent) and liking (benefit_eval_head)"
            " are functionally dissociable. Wanting produces persistent L1-directed"
            " approach after relocation; liking rapidly redirects to L2. Consistent"
            " with Berridge (1996) incentive salience / Culbreth et al. (2023) profiles."
        )
    elif criteria_met >= 2:
        interpretation = (
            "MECH-117 PARTIAL: Some dissociation signal present but below full threshold."
            " Consider longer training or adjusted goal_weight/decay_goal."
        )
    else:
        interpretation = (
            "MECH-117 NOT SUPPORTED: wanting/liking dissociation not demonstrated."
            " z_goal may not be receiving sufficient benefit_exposure updates,"
            " or goal_weight is too low relative to other trajectory scoring terms."
        )

    failure_notes: List[str] = []
    if not c1_pass:
        gap = r_wanting["resource_visit_rate"] - r_nogo["resource_visit_rate"]
        failure_notes.append(f"C1 FAIL: resource_rate gap={gap:.4f} < 0.04")
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: liking l2_redirect={r_liking['liking_l2_redirect_steps']}"
            " steps > 15"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: wanting l1_fraction={r_wanting['l1_directed_fraction']:.3f} < 0.25"
        )
    if not c4_pass:
        delta = r_wanting["harm_rate"] - r_nogo["harm_rate"]
        failure_notes.append(f"C4 FAIL: harm delta={delta:.4f} > 0.03")
    for n in failure_notes:
        print(f"  {n}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-074e -- MECH-117 Wanting/Liking Dissociation\n\n"
        f"**Status:** {status}\n**Claims:** MECH-112, MECH-117\n"
        f"**Seed:** {seed}  **Steps:** {TOTAL_STEPS}"
        f"  **Relocation:** step {RELOCATION_STEP}\n"
        f"**Fix from 074d:** CausalGridWorldV2 size=6, resource_respawn,"
        f" 50% proximity-following navigation (z_goal seeding fix).\n\n"
        f"## Conditions\n\n"
        f"- nogo: benefit_eval_enabled=False, z_goal_enabled=False\n"
        f"- liking: benefit_eval_enabled=True, z_goal_enabled=False\n"
        f"- wanting: benefit_eval_enabled=False, z_goal_enabled=True\n\n"
        f"## Results\n\n"
        f"| Metric | nogo | liking | wanting |\n|---|---|---|---|\n"
        f"| resource_visit_rate | {r_nogo['resource_visit_rate']:.4f}"
        f" | {r_liking['resource_visit_rate']:.4f}"
        f" | {r_wanting['resource_visit_rate']:.4f} |\n"
        f"| harm_rate | {r_nogo['harm_rate']:.4f}"
        f" | {r_liking['harm_rate']:.4f}"
        f" | {r_wanting['harm_rate']:.4f} |\n"
        f"| l2_redirect_steps | -- | {r_liking['liking_l2_redirect_steps']}"
        f" | -- |\n"
        f"| l1_directed_fraction | -- | -- | {r_wanting['l1_directed_fraction']:.3f} |\n"
        f"| goal_norm_final | -- | -- | {r_wanting['goal_norm_final']:.4f} |\n\n"
        f"## PASS Criteria\n\n| Criterion | Result |\n|---|---|\n"
        f"| C1: wanting resource >= nogo + 0.04 | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: liking l2_redirect <= 15 steps | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| C3: wanting l1_fraction >= 0.25 | {'PASS' if c3_pass else 'FAIL'} |\n"
        f"| C4: wanting harm delta <= 0.03 | {'PASS' if c4_pass else 'FAIL'} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n{interpretation}\n{failure_section}\n"
    )

    metrics = {
        "resource_rate_nogo":           float(r_nogo["resource_visit_rate"]),
        "resource_rate_liking":         float(r_liking["resource_visit_rate"]),
        "resource_rate_wanting":        float(r_wanting["resource_visit_rate"]),
        "harm_rate_nogo":               float(r_nogo["harm_rate"]),
        "harm_rate_liking":             float(r_liking["harm_rate"]),
        "harm_rate_wanting":            float(r_wanting["harm_rate"]),
        "liking_l2_redirect_steps":     float(r_liking["liking_l2_redirect_steps"]),
        "wanting_l1_directed_fraction": float(r_wanting["l1_directed_fraction"]),
        "wanting_goal_norm_final":      float(r_wanting["goal_norm_final"]),
        "wanting_goal_proximity_final": float(r_wanting["goal_proximity_final"]),
        "wanting_goal_active":          1.0 if r_wanting["goal_active"] else 0.0,
        "crit1_pass":                   1.0 if c1_pass else 0.0,
        "crit2_pass":                   1.0 if c2_pass else 0.0,
        "crit3_pass":                   1.0 if c3_pass else 0.0,
        "crit4_pass":                   1.0 if c4_pass else 0.0,
        "criteria_met":                 float(criteria_met),
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
            "nogo": r_nogo,
            "liking": r_liking,
            "wanting": r_wanting,
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
