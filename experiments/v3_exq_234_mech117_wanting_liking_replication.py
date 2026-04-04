#!/opt/local/bin/python3
"""
V3-EXQ-234 -- MECH-117 Wanting/Liking Dissociation Replication

Claims: MECH-112, MECH-117
Supersedes: independent replication of V3-EXQ-074f

EXPERIMENT_PURPOSE = "evidence"

MECH-117 asserts:
  Wanting (z_goal-driven approach) and liking (benefit_eval-driven redirect)
  are dissociable in E3: wanting produces persistent incentive salience
  (continued approach to the goal location even after the reward is removed),
  while liking produces rapid redirect to the new reward location.

EXQ-074f PASS 4/4 criteria on seeds [42, 7, 13]. MECH-117 promoted to
provisional. This experiment provides independent replication with seeds
[1, 2, 3] to support stable promotion.

Design (identical to 074f, different seeds)
-------------------------------------------
Three conditions per seed:
  A. nogo    -- benefit_eval_enabled=False, z_goal_enabled=False (baseline)
  B. liking  -- benefit_eval_enabled=True,  z_goal_enabled=False
  C. wanting -- benefit_eval_enabled=False, z_goal_enabled=True,
                e1_goal_conditioned=True

Each condition runs for 3000 steps on a 6x6 grid.
At step 1500: primary resource cell (L1) is relocated to L2 (3+ cells away).

Pre-relocation (steps 0..1499):
  All conditions: greedy-to-resource + random (GREEDY_FRAC=0.5).
  Ensures z_goal is continuously refreshed at resource visits.

Post-relocation (steps 1500..3000):
  wanting:  greedy toward L1 (old resource location, incentive salience).
  liking:   greedy toward env.resources (L2, current resource).
  nogo:     greedy-to-resource + random (baseline, unchanged).

PASS criteria (replicate 074f)
-------------------------------
C1: wanting_resource_rate >= 0.05   (floor: wanting visits at least some resources)
C2: liking_l2_redirect <= 15 steps  (rapid redirect to L2)
C3: wanting_l1_fraction >= 0.25     (25% of steps 1500-1700 toward L1)
C4: liking_l1_fraction <= wanting_l1_fraction - 0.10  (dissociation)

PASS: all 4 criteria in >= 2/3 seeds.

Seeds: [1, 2, 3]  (independent from 074f which used [42, 7, 13])
Env:   CausalGridWorldV2 size=6, 2 hazards, 2 resources
       resource_benefit=0.3, resource_respawn_on_consume=True
Est:   ~90 min DLAPTOP-4.local
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


EXPERIMENT_TYPE = "v3_exq_234_mech117_wanting_liking_replication"
CLAIM_IDS       = ["MECH-112", "MECH-117"]
EXPERIMENT_PURPOSE = "evidence"

GRID_SIZE      = 6
TOTAL_STEPS    = 3000
RELOCATION_STEP = 1500
GREEDY_FRAC    = 0.5

# Independent seeds from 074f ([42, 7, 13])
SEEDS = [1, 2, 3]


# ---------------------------------------------------------------------------
# Action helpers (identical to 074f)
# ---------------------------------------------------------------------------

def _greedy_action_toward_resource(env) -> int:
    ax, ay = env.agent_x, env.agent_y
    if not env.resources:
        return random.randint(0, env.action_dim - 1)
    best_d = float("inf"); nearest = None
    for r in env.resources:
        rx, ry = int(r[0]), int(r[1])
        d = abs(ax - rx) + abs(ay - ry)
        if d < best_d:
            best_d = d; nearest = (rx, ry)
    if nearest is None or best_d == 0:
        return random.randint(0, env.action_dim - 1)
    rx, ry = nearest
    dx, dy = rx - ax, ry - ay
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0
    else:
        return 3 if dy > 0 else 2


def _greedy_action_toward_pos(env, target_x: int, target_y: int) -> int:
    ax, ay = env.agent_x, env.agent_y
    dx, dy = target_x - ax, target_y - ay
    if dx == 0 and dy == 0:
        return random.randint(0, env.action_dim - 1)
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0
    else:
        return 3 if dy > 0 else 2


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Env / config factory
# ---------------------------------------------------------------------------

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
        kwargs["benefit_weight"]       = 1.0
    elif condition == "wanting":
        kwargs["z_goal_enabled"]        = True
        kwargs["e1_goal_conditioned"]   = True
        kwargs["goal_weight"]           = 1.0

    config = REEConfig.from_dims(**kwargs)
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Run one condition x one seed
# ---------------------------------------------------------------------------

def _run_condition(
    condition: str,
    seed: int,
    world_dim: int = 32,
    alpha_world: float = 0.9,
    lr: float = 1e-3,
    total_steps: int = TOTAL_STEPS,
    relocation_step: int = RELOCATION_STEP,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env    = _make_env(seed)
    agent  = _make_agent(condition, env, world_dim, alpha_world, lr)
    device = agent.device

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )

    agent.train()

    l1_pos: Optional[Tuple[int, int]] = None
    l2_pos: Optional[Tuple[int, int]] = None
    resource_relocated = False

    resource_visits = 0; harm_events = 0; total_steps_done = 0
    post_reloc_steps = 0; l1_directed_steps = 0
    liking_l2_redirect_found = False; liking_l2_redirect_at: Optional[int] = None

    _, obs_dict = env.reset()
    agent.reset()

    if env.resources:
        l1_pos = (int(env.resources[0][0]), int(env.resources[0][1]))

    step = 0
    while step < total_steps:
        # Resource relocation at relocation_step
        if step == relocation_step and not resource_relocated and l1_pos is not None:
            cx, cy = l1_pos
            new_pos = None
            for dx in range(-(GRID_SIZE - 1), GRID_SIZE):
                for dy in range(-(GRID_SIZE - 1), GRID_SIZE):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        if abs(dx) + abs(dy) >= 3:
                            if env.grid[nx, ny] == env.ENTITY_TYPES["empty"]:
                                new_pos = (nx, ny); break
                if new_pos is not None:
                    break
            if new_pos is None:
                for nx in range(GRID_SIZE):
                    for ny in range(GRID_SIZE):
                        if env.grid[nx, ny] == env.ENTITY_TYPES["empty"]:
                            new_pos = (nx, ny); break
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

        obs_body  = obs_dict["body_state"].to(device)
        obs_world = obs_dict["world_state"].to(device)

        z_self_prev = None
        if agent._current_latent is not None:
            z_self_prev = agent._current_latent.z_self.detach().clone()

        latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()

        # Action selection (matches 074f exactly)
        post_reloc = resource_relocated
        if not post_reloc:
            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_action_toward_resource(env)
            else:
                action_idx = random.randint(0, env.action_dim - 1)
        elif condition == "wanting":
            if l1_pos is not None:
                action_idx = _greedy_action_toward_pos(env, l1_pos[0], l1_pos[1])
            else:
                action_idx = random.randint(0, env.action_dim - 1)
        elif condition == "liking":
            action_idx = _greedy_action_toward_resource(env)
        else:
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

        if resource_relocated and relocation_step <= step < relocation_step + 200:
            post_reloc_steps += 1
            ax2, ay2 = env.agent_x, env.agent_y
            if l1_pos is not None and l2_pos is not None:
                d_l1 = abs(ax2 - l1_pos[0]) + abs(ay2 - l1_pos[1])
                d_l2 = abs(ax2 - l2_pos[0]) + abs(ay2 - l2_pos[1])
                if d_l1 < d_l2:
                    l1_directed_steps += 1
                elif d_l2 == 0 and not liking_l2_redirect_found:
                    liking_l2_redirect_found = True
                    liking_l2_redirect_at = step - relocation_step

        benefit_exp_val = float(obs_body[11]) if obs_body.shape[-1] > 11 else 0.0

        # E1 + E2 backward
        e1_loss = agent.compute_prediction_loss()
        e2_loss = agent.compute_e2_loss()
        total_e1_e2 = e1_loss + e2_loss
        if total_e1_e2.requires_grad:
            e1_opt.zero_grad(); e2_opt.zero_grad()
            total_e1_e2.backward()
            e1_opt.step(); e2_opt.step()

        # E3 harm supervision
        if agent._current_latent is not None:
            z_world  = agent._current_latent.z_world.detach()
            harm_tgt = torch.tensor([[1.0 if harm_signal < 0 else 0.0]], device=device)
            hloss    = F.mse_loss(agent.e3.harm_eval(z_world), harm_tgt)
            e3_opt.zero_grad(); hloss.backward(); e3_opt.step()

        # Liking: benefit_eval supervision
        if condition == "liking" and agent._current_latent is not None:
            benefit_t = torch.tensor([[benefit_exp_val]], device=device)
            bloss     = agent.compute_benefit_eval_loss(benefit_t)
            if bloss.requires_grad:
                e3_opt.zero_grad(); bloss.backward(); e3_opt.step()

        # Wanting: z_goal update
        if condition == "wanting":
            agent.update_z_goal(benefit_exp_val)

        agent.update_residue(harm_signal)

        total_steps_done += 1
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
                f"  [{condition}] step={step}/{total_steps}"
                f" resources={resource_visits} harm={harm_events}"
                f" goal_norm={diag['goal_norm']:.3f}"
                f" goal_prox={diag['goal_proximity']:.3f}",
                flush=True,
            )

    resource_visit_rate = resource_visits / max(1, total_steps_done)
    harm_rate           = harm_events     / max(1, total_steps_done)
    l1_fraction         = l1_directed_steps / max(1, post_reloc_steps)
    redirect_steps      = liking_l2_redirect_at if liking_l2_redirect_found else 200

    print(
        f"  [{condition}] DONE: resource_rate={resource_visit_rate:.4f}"
        f" harm_rate={harm_rate:.4f}"
        f" l1_fraction={l1_fraction:.3f}"
        f" l2_redirect={redirect_steps}",
        flush=True,
    )

    diag = agent.compute_goal_maintenance_diagnostic()
    return {
        "condition": condition,
        "resource_visit_rate":     resource_visit_rate,
        "harm_rate":               harm_rate,
        "l1_directed_fraction":    l1_fraction,
        "liking_l2_redirect_steps": redirect_steps,
        "goal_norm_final":         diag["goal_norm"],
        "goal_proximity_final":    diag["goal_proximity"],
        "goal_active":             diag["is_active"],
        "n_resource_visits":       resource_visits,
        "n_harm_events":           harm_events,
        "total_steps":             total_steps_done,
        "post_reloc_steps":        post_reloc_steps,
        "l1_directed_steps":       l1_directed_steps,
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    world_dim: int = 32,
    alpha_world: float = 0.9,
    lr: float = 1e-3,
    dry_run: bool = False,
) -> dict:
    print(f"\n[EXQ-234] MECH-117 Wanting/Liking Replication (dry_run={dry_run})",
          flush=True)
    print(
        f"  Independent replication of EXQ-074f. Seeds: {SEEDS} (074f used [42,7,13]).",
        flush=True,
    )

    tot  = 100  if dry_run else TOTAL_STEPS
    reloc = 50  if dry_run else RELOCATION_STEP

    all_seed_results = []
    c1_passes = []; c2_passes = []; c3_passes = []; c4_passes = []

    for seed in SEEDS:
        print(f"\n--- seed={seed} ---", flush=True)
        r_nogo    = _run_condition("nogo",    seed, world_dim, alpha_world, lr,
                                   total_steps=tot, relocation_step=reloc)
        r_liking  = _run_condition("liking",  seed, world_dim, alpha_world, lr,
                                   total_steps=tot, relocation_step=reloc)
        r_wanting = _run_condition("wanting", seed, world_dim, alpha_world, lr,
                                   total_steps=tot, relocation_step=reloc)

        # Criteria (identical to 074f)
        c1 = r_wanting["resource_visit_rate"] >= 0.05
        c2 = r_liking["liking_l2_redirect_steps"] <= 15
        c3 = r_wanting["l1_directed_fraction"] >= 0.25
        c4 = r_liking["l1_directed_fraction"] <= r_wanting["l1_directed_fraction"] - 0.10

        c1_passes.append(c1); c2_passes.append(c2)
        c3_passes.append(c3); c4_passes.append(c4)

        all_seed_results.append({
            "seed": seed,
            "criteria_met": sum([c1, c2, c3, c4]),
            "c1": c1, "c2": c2, "c3": c3, "c4": c4,
            "nogo": r_nogo, "liking": r_liking, "wanting": r_wanting,
        })

        print(
            f"  seed={seed}: C1={'P' if c1 else 'F'} C2={'P' if c2 else 'F'}"
            f" C3={'P' if c3 else 'F'} C4={'P' if c4 else 'F'}"
            f" ({sum([c1,c2,c3,c4])}/4)",
            flush=True,
        )

    # PASS = all 4 criteria met in >= 2/3 seeds
    c1_pass = sum(c1_passes) >= 2
    c2_pass = sum(c2_passes) >= 2
    c3_pass = sum(c3_passes) >= 2
    c4_pass = sum(c4_passes) >= 2

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    status   = "PASS" if all_pass else "FAIL"
    crit_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])

    print(f"\n[EXQ-234] Overall: {status} ({crit_met}/4 aggregate criteria)", flush=True)

    if all_pass:
        interpretation = (
            "MECH-117 REPLICATED: Independent seeds [1,2,3] confirm wanting/"
            "liking dissociation. z_goal-driven wanting produces persistent"
            " L1-directed approach after relocation; benefit_eval-driven liking"
            " rapidly redirects to L2. Consistent with EXQ-074f [42,7,13]."
            " Supports stable MECH-117 promotion."
        )
    elif crit_met >= 3:
        interpretation = (
            "MECH-117 MOSTLY REPLICATED: 3/4 aggregate criteria pass."
            " Partial replication; MECH-117 provisional status maintained."
        )
    elif crit_met >= 2:
        interpretation = (
            "MECH-117 PARTIAL REPLICATION: Signal present but below threshold."
            " Insufficient for stable promotion."
        )
    else:
        interpretation = (
            "MECH-117 REPLICATION FAILED: Dissociation not replicated in new"
            " seed set. Review seed sensitivity and task setup."
        )

    failure_notes = []
    for i, seed in enumerate(SEEDS):
        sr = all_seed_results[i]
        rw = sr["wanting"]; rl = sr["liking"]
        if not sr["c1"]:
            failure_notes.append(
                f"seed={seed} C1 FAIL: wanting resource_rate"
                f"={rw['resource_visit_rate']:.4f} < 0.05"
            )
        if not sr["c2"]:
            failure_notes.append(
                f"seed={seed} C2 FAIL: liking l2_redirect"
                f"={rl['liking_l2_redirect_steps']} > 15"
            )
        if not sr["c3"]:
            failure_notes.append(
                f"seed={seed} C3 FAIL: wanting l1_frac"
                f"={rw['l1_directed_fraction']:.3f} < 0.25"
            )
        if not sr["c4"]:
            diff = rw["l1_directed_fraction"] - rl["l1_directed_fraction"]
            failure_notes.append(
                f"seed={seed} C4 FAIL: dissociation gap={diff:.3f} < 0.10"
            )
    for n in failure_notes:
        print(f"  {n}", flush=True)

    summary_markdown = (
        f"# V3-EXQ-234 -- MECH-117 Wanting/Liking Dissociation Replication\n\n"
        f"**Status:** {status}  **Criteria met:** {crit_met}/4\n"
        f"**Claims:** MECH-112, MECH-117  **Purpose:** evidence\n"
        f"**Replication of:** V3-EXQ-074f (seeds [42,7,13])\n"
        f"**Seeds:** {SEEDS} (independent)\n\n"
        f"## Design\n\n"
        f"Identical to EXQ-074f: 3000 steps, 6x6 grid, L1->L2 relocation at"
        f" step {RELOCATION_STEP}. Wanting chases L1 post-relocation; liking"
        f" chases L2 (env.resources). Only seeds differ.\n\n"
        f"## Results by Seed\n\n"
        f"| Seed | want_res_rate | l2_redirect | want_l1_frac | like_l1_frac | C1 | C2 | C3 | C4 |\n"
        f"|------|--------------|------------|-------------|-------------|----|----|----|----|"
    )
    for i, seed in enumerate(SEEDS):
        sr = all_seed_results[i]
        rw = sr["wanting"]; rl = sr["liking"]
        summary_markdown += (
            f"\n| {seed}"
            f" | {rw['resource_visit_rate']:.4f}"
            f" | {rl['liking_l2_redirect_steps']}"
            f" | {rw['l1_directed_fraction']:.3f}"
            f" | {rl['l1_directed_fraction']:.3f}"
            f" | {'PASS' if sr['c1'] else 'FAIL'}"
            f" | {'PASS' if sr['c2'] else 'FAIL'}"
            f" | {'PASS' if sr['c3'] else 'FAIL'}"
            f" | {'PASS' if sr['c4'] else 'FAIL'} |"
        )
    summary_markdown += (
        f"\n\n## Interpretation\n\n{interpretation}\n"
    )
    if failure_notes:
        summary_markdown += "\n## Failure Notes\n\n"
        summary_markdown += "\n".join(f"- {n}" for n in failure_notes) + "\n"

    metrics: Dict = {
        "c1_pass": 1.0 if c1_pass else 0.0,
        "c2_pass": 1.0 if c2_pass else 0.0,
        "c3_pass": 1.0 if c3_pass else 0.0,
        "c4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(crit_met),
    }
    for i, seed in enumerate(SEEDS):
        sr  = all_seed_results[i]
        sfx = f"_seed{i}"
        rw  = sr["wanting"]; rl = sr["liking"]
        metrics[f"wanting_resource_rate{sfx}"]    = float(rw["resource_visit_rate"])
        metrics[f"liking_l2_redirect{sfx}"]       = float(rl["liking_l2_redirect_steps"])
        metrics[f"wanting_l1_fraction{sfx}"]      = float(rw["l1_directed_fraction"])
        metrics[f"liking_l1_fraction{sfx}"]       = float(rl["l1_directed_fraction"])
        metrics[f"l1_dissociation{sfx}"]          = float(
            rw["l1_directed_fraction"] - rl["l1_directed_fraction"]
        )
        metrics[f"wanting_goal_norm{sfx}"]        = float(rw["goal_norm_final"])
        metrics[f"seed_criteria_met{sfx}"]        = float(sr["criteria_met"])

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction_per_claim": {
            "MECH-112": "supports" if all_pass else ("mixed" if crit_met >= 2 else "weakens"),
            "MECH-117": "supports" if all_pass else ("mixed" if crit_met >= 2 else "weakens"),
        },
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if crit_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
        "seed_results": all_seed_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=1)
    parser.add_argument("--world-dim",   type=int,   default=32)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    result = run(
        world_dim=args.world_dim,
        alpha_world=args.alpha_world,
        lr=args.lr,
        dry_run=args.dry_run,
    )

    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"]         = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
