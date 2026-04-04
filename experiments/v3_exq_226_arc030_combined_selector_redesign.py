#!/opt/local/bin/python3
"""
V3-EXQ-226 -- ARC-030: COMBINED Selector Redesign

Claims: ARC-030
EXPERIMENT_PURPOSE = "evidence"

Scientific question: Does a COMBINED harm+goal CEM-style selector outperform
a HARM_ONLY selector on resource collection in a hazard-bearing goal task,
without excessive harm increase?

Context:
  EXQ-183 FAIL and EXQ-138a FAIL -- previous failures may have been due to
  z_goal not seeding reliably OR goal weight overwhelming harm avoidance.
  Precondition check: confirm z_goal_norm > 0.1 in COMBINED arm after warmup.
  If not met, report as substrate_limitation.

Design:
  - COMBINED: harm+goal CEM scoring (harm_eval - benefit_eval active, z_goal active).
    score = harm_eval(z_next) - 0.5 * benefit_eval(z_next) - 0.3 * goal_proximity(z_next).
    z_goal enabled, drive_weight=2.0 (SD-012).
  - HARM_ONLY: harm-only CEM scoring. z_goal ablated (z_goal_enabled=False,
    benefit_eval_enabled=False).
  - 3 seeds x 300 warmup eps + 300 eval eps x 200 steps.

Pre-registered PASS criteria:
  COMBINED improves resource_collection_rate or mean_benefit over HARM_ONLY by >= 20%:
    benefit_ratio = combined_benefit_rate / harm_only_benefit_rate >= 1.2
    OR
    mean_benefit_ratio = combined_mean_benefit / harm_only_mean_benefit >= 1.2
  AND
  harm_ratio = combined_harm_rate / harm_only_harm_rate <= 1.3

  Precondition (diagnostic, not gating PASS):
    z_goal_norm_combined > 0.1 after warmup.
    If not met: report outcome as substrate_limitation.

Decision:
  ALL PASS -> retain_ree (ARC-030 supported)
  precondition not met -> substrate_limitation
  PASS harm but FAIL benefit ratio -> hybridize
  FAIL harm ratio -> retire_ree_claim
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

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_226_arc030_combined_selector_redesign"
CLAIM_IDS = ["ARC-030"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
BENEFIT_RATIO_THRESH = 1.2   # COMBINED must outperform HARM_ONLY by >= 20%
HARM_RATIO_MAX       = 1.3   # COMBINED harm_rate / HARM_ONLY harm_rate <= 1.3
GOAL_NORM_PRECOND    = 0.1   # z_goal_norm must exceed this (precondition)

# Selector weights
HARM_WEIGHT    = 1.0
BENEFIT_WEIGHT = 0.5
GOAL_WEIGHT    = 0.3

# Episode settings
GRID_SIZE       = 10
WARMUP_EPISODES = 300
EVAL_EPISODES   = 300
STEPS_PER_EP    = 200
SEEDS           = [42, 7, 13]
MAX_BUF         = 4000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _manhattan_to_nearest_resource(env) -> int:
    if not env.resources:
        return 999
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
    else:
        return 3 if dy > 0 else 2


def _select_combined(agent: REEAgent, z_world: torch.Tensor, n_actions: int) -> int:
    """COMBINED: score = harm - benefit_weight*benefit - goal_weight*goal_prox."""
    with torch.no_grad():
        best_idx   = 0
        best_score = float("inf")
        for idx in range(n_actions):
            a_oh  = _onehot(idx, n_actions, z_world.device)
            z_next = agent.e2.world_forward(z_world, a_oh)
            harm   = agent.e3.harm_eval(z_next).mean().item()
            bene   = agent.e3.benefit_eval(z_next).mean().item()
            gprox  = agent.goal_state.goal_proximity(z_next).mean().item()
            score  = (
                HARM_WEIGHT * harm
                - BENEFIT_WEIGHT * bene
                - GOAL_WEIGHT * gprox
            )
            if score < best_score:
                best_score = score
                best_idx   = idx
    return best_idx


def _select_harm_only(agent: REEAgent, z_world: torch.Tensor, n_actions: int) -> int:
    """HARM_ONLY: score = harm_eval(z_next)."""
    with torch.no_grad():
        best_idx   = 0
        best_score = float("inf")
        for idx in range(n_actions):
            a_oh   = _onehot(idx, n_actions, z_world.device)
            z_next  = agent.e2.world_forward(z_world, a_oh)
            score   = agent.e3.harm_eval(z_next).mean().item()
            if score < best_score:
                best_score = score
                best_idx   = idx
    return best_idx


# ---------------------------------------------------------------------------
# Single seed run
# ---------------------------------------------------------------------------

def _run_seed(
    seed: int,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict:
    results = {}

    for condition in ["COMBINED", "HARM_ONLY"]:
        torch.manual_seed(seed)
        random.seed(seed)

        combined = (condition == "COMBINED")

        env = CausalGridWorldV2(
            seed=seed,
            size=GRID_SIZE,
            num_hazards=3,
            num_resources=4,
            hazard_harm=0.02,
            env_drift_interval=5,
            env_drift_prob=0.1,
            proximity_harm_scale=0.3,
            proximity_benefit_scale=0.18,
            proximity_approach_threshold=0.15,
            hazard_field_decay=0.5,
            energy_decay=0.005,
            use_proxy_fields=True,
            resource_respawn_on_consume=True,
        )
        n_actions = env.action_dim

        config = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=n_actions,
            self_dim=16,
            world_dim=32,
            alpha_world=0.9,
            alpha_self=0.3,
            reafference_action_dim=0,
            novelty_bonus_weight=0.0,
            benefit_eval_enabled=combined,
            benefit_weight=BENEFIT_WEIGHT if combined else 0.0,
            z_goal_enabled=combined,
            e1_goal_conditioned=combined,
            goal_weight=GOAL_WEIGHT if combined else 0.0,
            drive_weight=2.0 if combined else 0.0,
        )
        agent = REEAgent(config)

        std_params = [
            p for n, p in agent.named_parameters()
            if "harm_eval_head" not in n and "benefit_eval_head" not in n
        ]
        optimizer     = optim.Adam(std_params, lr=1e-3)
        harm_eval_opt = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=1e-4)
        benefit_eval_opt = (
            optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=1e-3)
            if combined else None
        )

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
                obs_harm  = obs_dict.get("harm_obs", None)

                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                agent.clock.advance()
                z_world_curr = latent.z_world.detach()

                # 50% greedy for COMBINED to seed z_goal
                if combined and random.random() < 0.5:
                    action_idx = _greedy_toward_resource(env)
                else:
                    action_idx = random.randint(0, n_actions - 1)
                action_oh = _onehot(action_idx, n_actions, agent.device)
                agent._last_action = action_oh

                dist   = _manhattan_to_nearest_resource(env)
                is_near = 1.0 if dist <= 2 else 0.0

                _, harm_signal, done, info, obs_dict = env.step(action_oh)

                b_exp = 0.0
                if obs_body.dim() == 1 and obs_body.shape[0] > 11:
                    b_exp = float(obs_body[11].item())
                elif obs_body.dim() > 1 and obs_body.shape[-1] > 11:
                    b_exp = float(obs_body[0, 11].item())

                # Standard training
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                total   = e1_loss + e2_loss
                if total.requires_grad:
                    optimizer.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

                # harm_eval (stratified)
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
                    pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                    ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                    zw_b = torch.cat(
                        [harm_buf_pos[i] for i in pi] +
                        [harm_buf_neg[i] for i in ni], dim=0
                    )
                    tgt = torch.cat([
                        torch.ones(k_p, 1, device=agent.device),
                        torch.zeros(k_n, 1, device=agent.device),
                    ], dim=0)
                    pred_h = agent.e3.harm_eval(zw_b)
                    hloss  = F.binary_cross_entropy(pred_h, tgt)
                    if hloss.requires_grad:
                        harm_eval_opt.zero_grad()
                        hloss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_head.parameters(), 0.5
                        )
                        harm_eval_opt.step()

                # benefit_eval (COMBINED only)
                if combined:
                    benefit_buf_zw.append(z_world_curr)
                    benefit_buf_lbl.append(is_near)
                    if len(benefit_buf_zw) > MAX_BUF:
                        benefit_buf_zw  = benefit_buf_zw[-MAX_BUF:]
                        benefit_buf_lbl = benefit_buf_lbl[-MAX_BUF:]

                    if len(benefit_buf_zw) >= 32 and step_i % 4 == 0:
                        k = min(32, len(benefit_buf_zw))
                        idxs = random.sample(range(len(benefit_buf_zw)), k)
                        zw_b = torch.cat([benefit_buf_zw[i] for i in idxs], dim=0)
                        lbl  = torch.tensor(
                            [benefit_buf_lbl[i] for i in idxs],
                            dtype=torch.float32
                        ).unsqueeze(1).to(agent.device)
                        pred_b = agent.e3.benefit_eval(zw_b)
                        bloss  = F.binary_cross_entropy(pred_b, lbl)
                        if bloss.requires_grad:
                            benefit_eval_opt.zero_grad()
                            bloss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                agent.e3.benefit_eval_head.parameters(), 0.5
                            )
                            benefit_eval_opt.step()
                            agent.e3.record_benefit_sample(k)

                    agent.update_z_goal(b_exp)

                if done:
                    break

            if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
                diag = agent.compute_goal_maintenance_diagnostic()
                print(
                    f"  [warmup] seed={seed} cond={condition}"
                    f" ep {ep+1}/{warmup_episodes}"
                    f" harm_pos={len(harm_buf_pos)}"
                    f" goal_norm={diag['goal_norm']:.3f}",
                    flush=True,
                )

        diag_final = agent.compute_goal_maintenance_diagnostic()
        goal_norm_f = float(diag_final["goal_norm"])
        precond_ok  = (not combined) or (goal_norm_f >= GOAL_NORM_PRECOND)

        if combined and not precond_ok:
            print(
                f"  [WARNING] seed={seed} cond={condition}"
                f" precond FAILED: goal_norm={goal_norm_f:.4f} < {GOAL_NORM_PRECOND}"
                f" -- reporting substrate_limitation",
                flush=True,
            )

        # ---- EVAL ----
        agent.eval()

        resource_counts: List[int]   = []
        mean_benefits:   List[float] = []
        harm_rates:      List[float] = []

        for _ in range(eval_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            ep_resources = 0
            ep_benefit   = 0.0
            ep_harm      = 0.0
            ep_steps     = 0

            for _ in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                obs_harm  = obs_dict.get("harm_obs", None)

                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                    agent.clock.advance()
                    z_world_curr = latent.z_world.detach()

                if combined:
                    action_idx = _select_combined(agent, z_world_curr, n_actions)
                else:
                    action_idx = _select_harm_only(agent, z_world_curr, n_actions)

                action_oh = _onehot(action_idx, n_actions, agent.device)
                agent._last_action = action_oh

                _, harm_signal, done, info, obs_dict = env.step(action_oh)
                ttype = info.get("transition_type", "none")
                if ttype == "resource":
                    ep_resources += 1

                b_exp = 0.0
                if obs_body.dim() == 1 and obs_body.shape[0] > 11:
                    b_exp = float(obs_body[11].item())
                elif obs_body.dim() > 1 and obs_body.shape[-1] > 11:
                    b_exp = float(obs_body[0, 11].item())
                ep_benefit += b_exp

                if float(harm_signal) < 0:
                    ep_harm += abs(float(harm_signal))
                ep_steps += 1
                if done:
                    break

            resource_counts.append(1 if ep_resources >= 1 else 0)
            mean_benefits.append(ep_benefit / max(1, ep_steps))
            harm_rates.append(ep_harm    / max(1, ep_steps))

        resource_rate = float(sum(resource_counts)) / max(1, len(resource_counts))
        mean_benefit  = float(sum(mean_benefits))   / max(1, len(mean_benefits))
        harm_rate     = float(sum(harm_rates))       / max(1, len(harm_rates))

        print(
            f"  [eval] seed={seed} cond={condition}"
            f" resource_rate={resource_rate:.3f}"
            f" mean_benefit={mean_benefit:.5f}"
            f" harm_rate={harm_rate:.5f}"
            f" goal_norm={goal_norm_f:.3f}",
            flush=True,
        )

        results[condition] = {
            "resource_rate": resource_rate,
            "mean_benefit":  mean_benefit,
            "harm_rate":     harm_rate,
            "goal_norm":     goal_norm_f,
            "precond_ok":    precond_ok,
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
    steps  = 20   if args.dry_run else STEPS_PER_EP
    seeds  = [42] if args.dry_run else SEEDS

    print(
        f"[V3-EXQ-226] ARC-030 COMBINED Selector Redesign"
        f" dry_run={args.dry_run}",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        print(f"\n[V3-EXQ-226] seed={seed}", flush=True)
        res = _run_seed(
            seed=seed,
            warmup_episodes=warmup,
            eval_episodes=n_eval,
            steps_per_episode=steps,
        )
        all_results.append({"seed": seed, "conditions": res})

    # ---- Aggregate ----
    def _avg(key: str, cond: str) -> float:
        return sum(r["conditions"][cond][key] for r in all_results) / max(1, len(all_results))

    comb_resource  = _avg("resource_rate", "COMBINED")
    ho_resource    = _avg("resource_rate", "HARM_ONLY")
    comb_benefit   = _avg("mean_benefit",  "COMBINED")
    ho_benefit     = _avg("mean_benefit",  "HARM_ONLY")
    comb_harm      = _avg("harm_rate",     "COMBINED")
    ho_harm        = _avg("harm_rate",     "HARM_ONLY")
    avg_goal_norm  = _avg("goal_norm",     "COMBINED")

    benefit_ratio      = comb_resource / max(1e-9, ho_resource)
    mean_benefit_ratio = comb_benefit  / max(1e-9, ho_benefit)
    harm_ratio         = comb_harm     / max(1e-9, ho_harm)

    precond_ok_all = all(
        r["conditions"]["COMBINED"]["precond_ok"] for r in all_results
    )

    if not precond_ok_all:
        outcome   = "FAIL"
        direction = "non_contributory"
        decision  = "substrate_limitation"
        benefit_pass = False
        harm_pass    = False
    else:
        benefit_pass = (
            benefit_ratio >= BENEFIT_RATIO_THRESH
            or mean_benefit_ratio >= BENEFIT_RATIO_THRESH
        )
        harm_pass    = harm_ratio <= HARM_RATIO_MAX

        if benefit_pass and harm_pass:
            outcome   = "PASS"
            direction = "supports"
            decision  = "retain_ree"
        elif harm_pass and not benefit_pass:
            outcome   = "FAIL"
            direction = "does_not_support"
            decision  = "hybridize"
        else:
            outcome   = "FAIL"
            direction = "weakens"
            decision  = "retire_ree_claim"

    print(f"\n[V3-EXQ-226] Results:", flush=True)
    print(
        f"  COMBINED:  resource_rate={comb_resource:.3f} mean_benefit={comb_benefit:.5f}"
        f" harm_rate={comb_harm:.5f}",
        flush=True,
    )
    print(
        f"  HARM_ONLY: resource_rate={ho_resource:.3f} mean_benefit={ho_benefit:.5f}"
        f" harm_rate={ho_harm:.5f}",
        flush=True,
    )
    print(
        f"  benefit_ratio={benefit_ratio:.2f}x mean_benefit_ratio={mean_benefit_ratio:.2f}x"
        f" harm_ratio={harm_ratio:.2f}x avg_goal_norm={avg_goal_norm:.4f}",
        flush=True,
    )
    print(
        f"  precond_ok={precond_ok_all}"
        f"  benefit_pass={benefit_pass}  harm_pass={harm_pass}",
        flush=True,
    )
    print(f"  -> {outcome} decision={decision}", flush=True)

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    ts      = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    manifest = {
        "run_id":               f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":      EXPERIMENT_TYPE,
        "architecture_epoch":   "ree_hybrid_guardrails_v1",
        "claim_ids":            CLAIM_IDS,
        "experiment_purpose":   EXPERIMENT_PURPOSE,
        "outcome":              outcome,
        "evidence_direction":   direction,
        "decision":             decision,
        "timestamp":            ts,
        "seeds":                seeds,
        "warmup_episodes":      warmup,
        "eval_episodes":        n_eval,
        "steps_per_episode":    steps,
        "harm_weight":          HARM_WEIGHT,
        "benefit_weight":       BENEFIT_WEIGHT,
        "goal_weight":          GOAL_WEIGHT,
        "benefit_ratio_thresh": BENEFIT_RATIO_THRESH,
        "harm_ratio_max":       HARM_RATIO_MAX,
        "goal_norm_precond":    GOAL_NORM_PRECOND,
        # Metrics
        "comb_resource_rate":   float(comb_resource),
        "ho_resource_rate":     float(ho_resource),
        "comb_mean_benefit":    float(comb_benefit),
        "ho_mean_benefit":      float(ho_benefit),
        "comb_harm_rate":       float(comb_harm),
        "ho_harm_rate":         float(ho_harm),
        "benefit_ratio":        float(benefit_ratio),
        "mean_benefit_ratio":   float(mean_benefit_ratio),
        "harm_ratio":           float(harm_ratio),
        "avg_goal_norm":        float(avg_goal_norm),
        "precond_ok_all":       precond_ok_all,
        "benefit_pass":         benefit_pass,
        "harm_pass":            harm_pass,
        # Per-seed
        "per_seed_results": [
            {
                "seed":              r["seed"],
                "comb_resource":     r["conditions"]["COMBINED"]["resource_rate"],
                "ho_resource":       r["conditions"]["HARM_ONLY"]["resource_rate"],
                "comb_mean_benefit": r["conditions"]["COMBINED"]["mean_benefit"],
                "ho_mean_benefit":   r["conditions"]["HARM_ONLY"]["mean_benefit"],
                "comb_harm":         r["conditions"]["COMBINED"]["harm_rate"],
                "ho_harm":           r["conditions"]["HARM_ONLY"]["harm_rate"],
                "goal_norm":         r["conditions"]["COMBINED"]["goal_norm"],
                "precond_ok":        r["conditions"]["COMBINED"]["precond_ok"],
            }
            for r in all_results
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-226] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
