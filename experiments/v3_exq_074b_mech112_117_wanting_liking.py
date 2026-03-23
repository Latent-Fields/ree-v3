#!/opt/local/bin/python3
"""
V3-EXQ-074b -- MECH-117 Wanting/Liking Dissociation

Claims: MECH-112, MECH-117

Tests the functional dissociation between:
  - Liking (benefit_eval_head): learned from benefit_exposure at receipt;
    sharp proximity spike; redirects rapidly when resource moves.
  - Wanting (z_goal_latent): slow-decay attractor; continuous spatial
    gradient; persists after resource relocation (incentive salience).

Three conditions (matched seeds):
  A. nogo    -- benefit_eval_enabled=False, z_goal_enabled=False (baseline)
  B. liking  -- benefit_eval_enabled=True,  z_goal_enabled=False
  C. wanting -- benefit_eval_enabled=False, z_goal_enabled=True, e1_goal_conditioned=True

Each condition trains for 3000 steps on an 8x8 grid.
At step 1500: primary resource cell (L1) is relocated to L2 (3+ cells away).

PASS criteria (ALL required):
  C1: wanting_resource_rate >= nogo_resource_rate + 0.04
      (z_goal_enabled improves resource-visit rate)
  C2: liking_l2_redirect <= 15
      (liking re-targets to L2 within 15 steps of relocation)
  C3: wanting_l1_steps >= 0.25 * 200
      (wanting maintains approach toward L1 for >=25% of steps 1500-1700)
  C4: harm_wanting <= harm_nogo + 0.03
      (z_goal does not increase harm rate)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_074b_mech112_117_wanting_liking"
CLAIM_IDS = ["MECH-112", "MECH-117"]

BODY_OBS_DIM = 12   # use_proxy_fields=True
WORLD_OBS_DIM = 250
ACTION_DIM = 4
GRID_SIZE = 8
TOTAL_STEPS = 3000
RELOCATION_STEP = 1500


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=GRID_SIZE,
        num_resources=2,
        num_hazards=2,
        use_proxy_fields=True,
        seed=seed,
        resource_benefit=0.3,
        hazard_harm=0.02,
    )


def _make_agent(
    condition: str,
    world_dim: int = 32,
    alpha_world: float = 0.9,
    lr: float = 1e-3,
) -> Tuple[REEAgent, dict]:
    kwargs: dict = dict(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
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
    agent = REEAgent(config)
    return agent, kwargs


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
    agent, _ = _make_agent(condition, world_dim, alpha_world, lr)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
    if agent.goal_state is not None and hasattr(agent.e1, 'goal_input_proj'):
        if agent.e1.goal_input_proj is not None:
            e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)

    agent.train()

    # Track L1 (original resource cell) and L2 (new after relocation)
    l1_pos: Optional[Tuple[int, int]] = None
    l2_pos: Optional[Tuple[int, int]] = None
    resource_relocated = False

    # Metrics
    resource_visits = 0
    harm_events = 0
    total_steps = 0

    # Post-relocation tracking (steps 1500-1700)
    post_reloc_steps = 0
    l1_directed_steps = 0   # agent moves toward L1 (wanting stays)
    l2_redirect_steps = 0   # steps until agent reaches L2 post-reloc (liking redirect)
    liking_l2_redirect_found = False
    liking_l2_redirect_at: Optional[int] = None

    # Episode management: run flat steps (not episodes)
    _, obs_dict = env.reset()
    agent.reset()

    # Record initial resource positions
    if env.resources:
        l1_pos = (env.resources[0][0], env.resources[0][1])

    step = 0
    while step < TOTAL_STEPS:
        # Resource relocation at step 1500
        if step == RELOCATION_STEP and not resource_relocated and l1_pos is not None:
            # Find a new cell 3+ cells away from L1
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
                # fallback: pick any empty cell
                for nx in range(GRID_SIZE):
                    for ny in range(GRID_SIZE):
                        if env.grid[nx, ny] == env.ENTITY_TYPES["empty"]:
                            new_pos = (nx, ny)
                            break
                    if new_pos is not None:
                        break
            if new_pos is not None:
                l2_pos = new_pos
                # Remove L1 from grid, add L2
                if env.resources:
                    old_x, old_y = env.resources[0]
                    env.grid[old_x, old_y] = env.ENTITY_TYPES["empty"]
                    env.resources[0] = [l2_pos[0], l2_pos[1]]
                    env.grid[l2_pos[0], l2_pos[1]] = env.ENTITY_TYPES["resource"]
                    env._compute_proximity_fields()
                resource_relocated = True
                print(
                    f"  [{condition}] step={step}: L1=({cx},{cy}) -> L2=({l2_pos[0]},{l2_pos[1]})",
                    flush=True,
                )

        obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
        obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

        z_self_t = None
        if agent._current_latent is not None:
            z_self_t = agent._current_latent.z_self.detach().clone()

        latent = agent.sense(obs_body, obs_world)
        ticks = agent.clock.advance()
        e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
            1, world_dim, device=agent.device
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks, temperature=1.0)

        if z_self_t is not None:
            agent.record_transition(z_self_t, action, latent.z_self.detach())

        _, reward, done, info, obs_dict = env.step(action)
        harm_signal = float(reward) if reward < 0 else 0.0
        ttype = info.get("transition_type", "none")

        if ttype == "resource":
            resource_visits += 1
        if ttype in ("agent_caused_hazard", "hazard_approach"):
            harm_events += 1

        # Track L1/L2 directed steps after relocation
        if resource_relocated and RELOCATION_STEP <= step < RELOCATION_STEP + 200:
            post_reloc_steps += 1
            ax, ay = env.agent_x, env.agent_y
            if l1_pos is not None and l2_pos is not None:
                d_to_l1 = abs(ax - l1_pos[0]) + abs(ay - l1_pos[1])
                d_to_l2 = abs(ax - l2_pos[0]) + abs(ay - l2_pos[1])
                if d_to_l1 < d_to_l2:
                    l1_directed_steps += 1
                elif d_to_l2 == 0 and not liking_l2_redirect_found:
                    liking_l2_redirect_found = True
                    liking_l2_redirect_at = step - RELOCATION_STEP

        # Benefit exposure for wanted/liking conditions
        benefit_exp_val = float(obs_body[11])

        # E1 + E2 training (combined backward avoids inplace op conflict after e1_opt.step)
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
            harm_target = torch.tensor([[1.0 if harm_signal < 0 else 0.0]], device=agent.device)
            harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
            e3_opt.zero_grad()
            harm_loss.backward()
            e3_opt.step()

        # Liking: benefit_eval supervision
        if condition == "liking" and agent._current_latent is not None:
            benefit_target = torch.tensor([[benefit_exp_val]], device=agent.device)
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

        # Episode boundary: reset env but keep agent memory
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            # Re-apply resource relocation state to newly reset env
            if resource_relocated and l2_pos is not None:
                # The env.reset() restores resources; re-relocate
                if env.resources:
                    old_x, old_y = env.resources[0]
                    env.grid[old_x, old_y] = env.ENTITY_TYPES["empty"]
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
    print(f"\n[EXQ-074b] MECH-117 Wanting/Liking Dissociation seed={seed}", flush=True)

    r_nogo    = _run_condition("nogo",    seed, world_dim, alpha_world, lr)
    r_liking  = _run_condition("liking",  seed, world_dim, alpha_world, lr)
    r_wanting = _run_condition("wanting", seed, world_dim, alpha_world, lr)

    # PASS criteria
    # C1: wanting resource rate >= nogo + 0.04
    c1_pass = r_wanting["resource_visit_rate"] >= r_nogo["resource_visit_rate"] + 0.04

    # C2: liking redirects to L2 within 15 steps
    c2_pass = r_liking["liking_l2_redirect_steps"] <= 15

    # C3: wanting maintains L1 approach for >=25% of steps 1500-1700
    c3_pass = r_wanting["l1_directed_fraction"] >= 0.25

    # C4: wanting harm rate <= nogo + 0.03
    c4_pass = r_wanting["harm_rate"] <= r_nogo["harm_rate"] + 0.03

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[EXQ-074b] Results:", flush=True)
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
            f"C2 FAIL: liking l2_redirect={r_liking['liking_l2_redirect_steps']} steps > 15"
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
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = (
        f"# V3-EXQ-074b -- MECH-117 Wanting/Liking Dissociation\n\n"
        f"**Status:** {status}\n**Claims:** MECH-112, MECH-117\n"
        f"**Seed:** {seed}  **Steps:** {TOTAL_STEPS}"
        f"  **Relocation:** step {RELOCATION_STEP}\n\n"
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
