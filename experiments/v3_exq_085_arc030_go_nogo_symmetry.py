"""
V3-EXQ-085 -- ARC-030 Go/NoGo Symmetry (Approach-Avoidance)

Claims: ARC-030

ARC-030 asserts that the three BG-like loops require symmetric Go (approach) and
NoGo (avoidance) sub-channels. Pure NoGo architecture -- harm avoidance only --
produces behavioral flatness: the agent successfully avoids hazards but fails to
approach resources, resulting in near-zero benefit accumulation. Activating the
full Go channel (z_goal attractor + benefit_eval training) should break this
flatness: the agent navigates toward resources while maintaining hazard avoidance.

This is the first experiment to train the Go channel end-to-end.

Discriminative pair:
  NOGO_ONLY -- harm_eval trained; z_goal disabled; benefit_eval disabled
  GO_NOGO   -- harm_eval + z_goal updated every step + benefit_eval trained
               (full Go channel: wanting + liking both active, ARC-030 design)

Primary discriminator (pre-registered threshold >= 0.001):
  delta_benefit_rate = benefit_rate_GONOGO - benefit_rate_NOGO
  benefit_rate = cumulative benefit signal / total eval steps

PASS criteria (ALL required):
  C1: delta_benefit_rate > 0.001
      (Go channel generates measurable benefit advantage over NoGo-only)
  C2: harm_rate_GONOGO <= harm_rate_NOGO * 1.5
      (Go channel does not catastrophically impair harm avoidance)
  C3: benefit_rate_GONOGO > 0.001
      (Go agent actively collects benefit -- not still flat)
  C4: per-seed: benefit_rate_GONOGO > benefit_rate_NOGO for both seeds
      (consistent direction -- Go channel robustly improves approach)

Decision outcomes:
  retain_ree:       C1+C2+C3+C4 all pass
  hybridize:        C4 passes (consistent direction) but C1 or C3 marginal
  retire_ree_claim: C4 fails (no consistent benefit advantage from Go channel)

Biological reference: ARC-030 -- Frank 2005 (D1/D2 competition in BG),
Bariselli et al. 2018 (direct/indirect pathways tuned to same actions).

Architecture epoch: ree_hybrid_guardrails_v1
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_085_arc030_go_nogo_symmetry"
CLAIM_IDS = ["ARC-030"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.2,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.05,   # symmetric with harm scale (ARC-030)
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _run_single(
    seed: int,
    go_nogo: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    harm_scale: float,
    proximity_scale: float,
) -> Dict:
    """Run one (seed, condition) cell. Returns benefit/harm rate metrics."""
    cond_label = "GO_NOGO" if go_nogo else "NOGO_ONLY"
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.2,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        z_goal_enabled=go_nogo,
        benefit_eval_enabled=go_nogo,
        e1_goal_conditioned=go_nogo,
        reafference_action_dim=0,
    )
    agent = REEAgent(config)

    # Separate optimizers: standard params, harm head, benefit head
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n and "benefit_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    benefit_eval_opt = None
    if go_nogo:
        benefit_eval_params = list(agent.e3.benefit_eval_head.parameters())
        benefit_eval_opt = optim.Adam(benefit_eval_params, lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    benefit_buf_pos: List[torch.Tensor] = []  # z_world near resources
    benefit_buf_neg: List[torch.Tensor] = []  # z_world far from resources
    MAX_BUF = 2000

    print(
        f"  [train] {cond_label} seed={seed}"
        f" warmup={warmup_episodes} eval={eval_episodes}",
        flush=True,
    )

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------
    agent.train()
    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            benefit_exp = float(
                obs_body[11] if hasattr(obs_body, "__len__") else obs_body
            )

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, reward, done, info, obs_dict = env.step(action)
            reward_f = float(reward)
            harm_signal = reward_f if reward_f < 0 else 0.0
            benefit_signal = reward_f if reward_f > 0 else 0.0

            # --- Standard E1 + E2 ---
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_std = e1_loss + e2_loss
            if total_std.requires_grad:
                optimizer.zero_grad()
                total_std.backward()
                torch.nn.utils.clip_grad_norm_(standard_params, 1.0)
                optimizer.step()

            # --- Harm_eval (NoGo channel, both conditions) ---
            is_harm = reward_f < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k = min(16, len(harm_buf_pos), len(harm_buf_neg))
                zw_p = torch.cat(random.sample(harm_buf_pos, k), dim=0)
                zw_n = torch.cat(random.sample(harm_buf_neg, k), dim=0)
                zw_b = torch.cat([zw_p, zw_n], dim=0)
                tgt = torch.cat([
                    torch.ones(k, 1, device=agent.device),
                    torch.zeros(k, 1, device=agent.device),
                ], dim=0)
                h_loss = F.mse_loss(agent.e3.harm_eval(zw_b), tgt)
                if h_loss.requires_grad:
                    harm_eval_opt.zero_grad()
                    h_loss.backward()
                    harm_eval_opt.step()

            # --- Go channel (GO_NOGO condition only) ---
            if go_nogo:
                # Update z_goal attractor (wanting: MECH-112)
                agent.update_z_goal(benefit_exp)

                # Train benefit_eval_head (liking: MECH-112 Go sub-channel)
                is_benefit = reward_f > 0 or benefit_exp > 0.05
                if is_benefit:
                    benefit_buf_pos.append(z_world_curr)
                    if len(benefit_buf_pos) > MAX_BUF:
                        benefit_buf_pos = benefit_buf_pos[-MAX_BUF:]
                else:
                    benefit_buf_neg.append(z_world_curr)
                    if len(benefit_buf_neg) > MAX_BUF:
                        benefit_buf_neg = benefit_buf_neg[-MAX_BUF:]

                if len(benefit_buf_pos) >= 4 and len(benefit_buf_neg) >= 4:
                    k = min(16, len(benefit_buf_pos), len(benefit_buf_neg))
                    zw_p = torch.cat(random.sample(benefit_buf_pos, k), dim=0)
                    zw_n = torch.cat(random.sample(benefit_buf_neg, k), dim=0)
                    zw_b = torch.cat([zw_p, zw_n], dim=0)
                    tgt = torch.cat([
                        torch.ones(k, 1, device=agent.device),
                        torch.zeros(k, 1, device=agent.device),
                    ], dim=0)
                    b_loss = F.mse_loss(agent.e3.benefit_eval(zw_b), tgt)
                    if b_loss.requires_grad:
                        benefit_eval_opt.zero_grad()
                        b_loss.backward()
                        benefit_eval_opt.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            goal_str = ""
            if go_nogo and agent.goal_state is not None:
                goal_str = f" goal_norm={agent.goal_state.goal_norm():.3f}"
            print(
                f"  [train] {cond_label} seed={seed} ep {ep+1}/{warmup_episodes}"
                f"{goal_str}",
                flush=True,
            )

    # -------------------------------------------------------------------
    # Eval
    # -------------------------------------------------------------------
    agent.eval()
    benefit_accumulated: List[float] = []
    harm_accumulated: List[float] = []
    goal_proximity_vals: List[float] = []
    n_fatal = 0

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_benefit = 0.0
        ep_harm = 0.0
        ep_steps = 0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                e1_prior = agent._e1_tick(latent)
                candidates = agent.generate_trajectories(latent, e1_prior, {
                    "e1_tick": True, "e2_tick": True, "e3_tick": True,
                })
                action = agent.select_action(candidates, {
                    "e1_tick": True, "e2_tick": True, "e3_tick": True,
                }, temperature=0.5)

            action_idx = int(action.squeeze().argmax().item())
            agent._last_action = action

            _, reward, done, info, obs_dict = env.step(action)
            reward_f = float(reward)
            ep_benefit += max(0.0, reward_f)
            ep_harm += abs(min(0.0, reward_f))
            ep_steps += 1

            if go_nogo and agent.goal_state is not None and agent._current_latent is not None:
                try:
                    gp = float(
                        agent.goal_state.goal_proximity(
                            agent._current_latent.z_world
                        ).mean().item()
                    )
                    goal_proximity_vals.append(gp)
                except Exception:
                    n_fatal += 1

            if done:
                break

        if ep_steps > 0:
            benefit_accumulated.append(ep_benefit / ep_steps)
            harm_accumulated.append(ep_harm / ep_steps)

    benefit_rate = float(sum(benefit_accumulated) / max(1, len(benefit_accumulated)))
    harm_rate = float(sum(harm_accumulated) / max(1, len(harm_accumulated)))
    goal_proximity_mean = (
        float(sum(goal_proximity_vals) / max(1, len(goal_proximity_vals)))
        if goal_proximity_vals else float("nan")
    )

    print(
        f"  [eval] {cond_label} seed={seed}"
        f" benefit_rate={benefit_rate:.5f}"
        f" harm_rate={harm_rate:.5f}"
        f" goal_prox={goal_proximity_mean:.4f}"
        f" n_fatal={n_fatal}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "go_nogo": go_nogo,
        "benefit_rate": benefit_rate,
        "harm_rate": harm_rate,
        "goal_proximity_mean": goal_proximity_mean,
        "n_benefit_buf_pos": len(benefit_buf_pos) if go_nogo else 0,
        "n_harm_buf_pos": len(harm_buf_pos),
        "n_fatal": n_fatal,
    }


def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 150,
    eval_episodes: int = 30,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    """ARC-030: NOGO_ONLY vs GO_NOGO discriminative pair."""
    print(
        f"\n[EXQ-085] ARC-030 Go/NoGo symmetry"
        f" seeds={seeds} warmup={warmup_episodes} eval={eval_episodes}",
        flush=True,
    )

    results_go: List[Dict] = []
    results_nogo: List[Dict] = []

    for seed in seeds:
        for go_nogo in [False, True]:
            r = _run_single(
                seed=seed,
                go_nogo=go_nogo,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                harm_scale=harm_scale,
                proximity_scale=proximity_scale,
            )
            if go_nogo:
                results_go.append(r)
            else:
                results_nogo.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    benefit_rate_go   = _avg(results_go,   "benefit_rate")
    benefit_rate_nogo = _avg(results_nogo, "benefit_rate")
    harm_rate_go      = _avg(results_go,   "harm_rate")
    harm_rate_nogo    = _avg(results_nogo, "harm_rate")
    goal_prox_go      = _avg(results_go,   "goal_proximity_mean")
    delta_benefit     = benefit_rate_go - benefit_rate_nogo

    # Per-seed directionality (C4)
    c4_per_seed = [
        r_go["benefit_rate"] > r_nogo["benefit_rate"]
        for r_go, r_nogo in zip(results_go, results_nogo)
    ]
    c4_pass = all(c4_per_seed)

    # Pre-registered PASS criteria
    c1_pass = delta_benefit > 0.001
    c2_pass = harm_rate_go <= harm_rate_nogo * 1.5
    c3_pass = benefit_rate_go > 0.001

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c4_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[EXQ-085] Final results:", flush=True)
    print(
        f"  benefit_rate:  GO_NOGO={benefit_rate_go:.5f}"
        f"  NOGO_ONLY={benefit_rate_nogo:.5f}"
        f"  delta={delta_benefit:+.5f}",
        flush=True,
    )
    print(
        f"  harm_rate:     GO_NOGO={harm_rate_go:.5f}"
        f"  NOGO_ONLY={harm_rate_nogo:.5f}",
        flush=True,
    )
    print(
        f"  goal_prox_GO:  {goal_prox_go:.4f}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/4)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: delta_benefit_rate={delta_benefit:.5f} <= 0.001"
            " -- Go channel not producing measurable benefit advantage"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: harm_rate_GO={harm_rate_go:.5f}"
            f" > harm_rate_NOGO*1.5={harm_rate_nogo*1.5:.5f}"
            " -- Go channel impairs hazard avoidance"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: benefit_rate_GO={benefit_rate_go:.5f} <= 0.001"
            " -- Go agent still behaviourally flat (training too short or goal_weight too low)"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: inconsistent directionality across seeds -- {c4_per_seed}"
        )
    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "ARC-030 SUPPORTED: Activating the Go channel (z_goal + benefit_eval)"
            f" produces benefit_rate={benefit_rate_go:.5f} vs NOGO_ONLY"
            f" {benefit_rate_nogo:.5f} (delta={delta_benefit:+.5f} > 0.001)."
            " Hazard avoidance maintained. Pure NoGo produces measurably flatter"
            " resource-approaching behaviour. D1/D2 symmetry is required."
        )
    elif c4_pass:
        interpretation = (
            "ARC-030 PARTIAL: Consistent direction (Go > NoGo) across seeds but"
            " delta below threshold. Go channel is working but weakly."
            " Consider increasing goal_weight or proximity_scale, or longer warmup."
        )
    else:
        interpretation = (
            "ARC-030 NOT SUPPORTED: Go channel does not consistently improve"
            " resource collection. Either benefit_eval training is insufficient,"
            " z_goal attractor is not forming, or the environment benefit signal"
            " is too sparse. Diagnostic: check n_benefit_buf_pos in metrics."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_go_rows = "\n".join(
        f"  seed={r['seed']}: benefit_rate={r['benefit_rate']:.5f}"
        f" harm_rate={r['harm_rate']:.5f}"
        f" goal_prox={r['goal_proximity_mean']:.4f}"
        f" benefit_buf={r['n_benefit_buf_pos']}"
        for r in results_go
    )
    per_nogo_rows = "\n".join(
        f"  seed={r['seed']}: benefit_rate={r['benefit_rate']:.5f}"
        f" harm_rate={r['harm_rate']:.5f}"
        for r in results_nogo
    )

    summary_markdown = (
        f"# V3-EXQ-085 -- ARC-030 Go/NoGo Symmetry (Approach-Avoidance)\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-030\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: delta_benefit_rate          > 0.001\n"
        f"C2: harm_rate_GO                <= harm_rate_NOGO * 1.5\n"
        f"C3: benefit_rate_GO             > 0.001\n"
        f"C4: per-seed GO > NOGO direction\n\n"
        f"## Results\n\n"
        f"| Condition | benefit_rate | harm_rate | goal_proximity |\n"
        f"|-----------|-------------|-----------|----------------|\n"
        f"| GO_NOGO   | {benefit_rate_go:.5f}  | {harm_rate_go:.5f}"
        f" | {goal_prox_go:.4f} |\n"
        f"| NOGO_ONLY | {benefit_rate_nogo:.5f}"
        f" | {harm_rate_nogo:.5f}"
        f" | N/A |\n\n"
        f"**delta_benefit_rate (GO - NOGO): {delta_benefit:+.5f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|-----------|--------|-------|\n"
        f"| C1: delta_benefit_rate > 0.001       | {'PASS' if c1_pass else 'FAIL'}"
        f" | {delta_benefit:+.5f} |\n"
        f"| C2: harm_rate_GO <= NOGO*1.5         | {'PASS' if c2_pass else 'FAIL'}"
        f" | {harm_rate_go:.5f} vs {harm_rate_nogo*1.5:.5f} |\n"
        f"| C3: benefit_rate_GO > 0.001          | {'PASS' if c3_pass else 'FAIL'}"
        f" | {benefit_rate_go:.5f} |\n"
        f"| C4: per-seed direction               | {'PASS' if c4_pass else 'FAIL'}"
        f" | {c4_per_seed} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"GO_NOGO:\n{per_go_rows}\n\n"
        f"NOGO_ONLY:\n{per_nogo_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "benefit_rate_go":        float(benefit_rate_go),
        "benefit_rate_nogo":      float(benefit_rate_nogo),
        "delta_benefit_rate":     float(delta_benefit),
        "harm_rate_go":           float(harm_rate_go),
        "harm_rate_nogo":         float(harm_rate_nogo),
        "goal_proximity_go":      float(goal_prox_go),
        "n_benefit_buf_go":       float(sum(r["n_benefit_buf_pos"] for r in results_go)),
        "n_harm_buf_go":          float(sum(r["n_harm_buf_pos"] for r in results_go)),
        "n_harm_buf_nogo":        float(sum(r["n_harm_buf_pos"] for r in results_nogo)),
        "n_seeds":                float(len(seeds)),
        "alpha_world":            float(alpha_world),
        "proximity_scale":        float(proximity_scale),
        "crit1_pass":             1.0 if c1_pass else 0.0,
        "crit2_pass":             1.0 if c2_pass else 0.0,
        "crit3_pass":             1.0 if c3_pass else 0.0,
        "crit4_pass":             1.0 if c4_pass else 0.0,
        "criteria_met":           float(criteria_met),
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
        "fatal_error_count": sum(r["n_fatal"] for r in results_go + results_nogo),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 7])
    parser.add_argument("--warmup",          type=int,   default=150)
    parser.add_argument("--eval-eps",        type=int,   default=30)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
        proximity_scale=args.proximity_scale,
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
