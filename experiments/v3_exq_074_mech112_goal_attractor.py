"""
V3-EXQ-074 -- MECH-112 Goal Attractor (Benefit Eval Head)

Claims: MECH-112

MECH-112 asserts that a prospective goal drive (benefit_eval_head on E3,
supervised by resource proximity) increases resource-directed behaviour
compared to NoGo-only avoidance.

The key distinction from harm avoidance (reactive) is that benefit_eval
evaluates trajectory proposals BEFORE execution -- it is prospective.
The agent should orient toward resource-gradient cells even before arriving.

Two conditions (matched seeds):
  A. NoGoOnly  -- benefit_eval_enabled=False (no goal attractor)
  B. GoNoGo    -- benefit_eval_enabled=True, benefit_weight=1.0 (ARC-030)

Both conditions use CausalGridWorldV2 (use_proxy_fields=True) for access
to benefit_exposure (body_state[11]) as the benefit supervision signal.

PASS criteria (ALL required):
  C1: resource_visit_rate_gongo >= resource_visit_rate_nogo + 0.05
      (Go channel produces systematically higher resource contact)
  C2: benefit_eval_r2 > 0.10
      (benefit_eval_head learns to predict resource proximity)
  C3: pre_arrival_benefit_score_gongo > pre_arrival_benefit_score_nogo + 0.02
      (trajectories selected by GoNoGo point toward resources before arrival)
  C4: harm_rate_gongo <= harm_rate_nogo + 0.03
      (Go channel does not increase harm exposure)

Informational:
  benefit_exposure_mean per condition (environment exposure level)
  benefit_eval_calibration (mean predicted benefit at resource cells vs empty)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import math
from typing import Dict, List

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_074_mech112_goal_attractor"
CLAIM_IDS = ["MECH-112"]

BODY_OBS_DIM = 12   # use_proxy_fields=True
WORLD_OBS_DIM = 250
ACTION_DIM = 4


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=10,
        num_resources=5,
        num_hazards=3,
        use_proxy_fields=True,
        seed=seed,
    )


def _run_single(
    seed: int,
    benefit_eval_enabled: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    benefit_weight: float = 1.0,
) -> Dict:
    cond_label = "GONGO" if benefit_eval_enabled else "NOGO_ONLY"

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)

    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        benefit_eval_enabled=benefit_eval_enabled,
        benefit_weight=benefit_weight,
    )
    agent = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)

    print(f"\n[EXQ-074] TRAIN {cond_label} seed={seed}", flush=True)
    agent.train()

    # Training: benefit_eval_head supervised from benefit_exposure (body_state[11])
    benefit_predictions: List[float] = []  # track benefit prediction quality
    benefit_targets_train: List[float] = []

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
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
            ep_harm += abs(harm_signal)

            # E1 loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()

            # E2 loss
            e2_loss = agent.compute_e2_loss()
            if e2_loss.requires_grad:
                e2_opt.zero_grad()
                e2_loss.backward()
                e2_opt.step()

            # E3 harm supervision
            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
                e3_opt.zero_grad()
                harm_loss.backward()
                e3_opt.step()

            # Benefit eval supervision (GoNoGo only)
            if benefit_eval_enabled and agent._current_latent is not None:
                benefit_exp_val = float(obs_body[11])
                benefit_target = torch.tensor([[benefit_exp_val]], device=agent.device)
                benefit_loss = agent.compute_benefit_eval_loss(benefit_target)
                if benefit_loss.requires_grad:
                    e3_opt.zero_grad()
                    benefit_loss.backward()
                    e3_opt.step()

                # Track for R2 calculation (last 1000 steps of training)
                if ep >= warmup_episodes - 10:
                    z_world = agent._current_latent.z_world.detach()
                    with torch.no_grad():
                        pred = float(agent.e3.benefit_eval(z_world).item())
                    benefit_predictions.append(pred)
                    benefit_targets_train.append(benefit_exp_val)

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"  [train] {cond_label} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm={ep_harm:.3f}",
                flush=True,
            )

    # Compute benefit_eval R2 (GoNoGo condition only)
    benefit_eval_r2 = 0.0
    if benefit_eval_enabled and len(benefit_predictions) >= 5:
        preds = torch.tensor(benefit_predictions)
        targets = torch.tensor(benefit_targets_train)
        ss_res = ((preds - targets) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        benefit_eval_r2 = float(max(0.0, (1 - ss_res / (ss_tot + 1e-8)).item()))
    print(f"  [benefit_r2] {cond_label} r2={benefit_eval_r2:.4f}", flush=True)

    # --- EVAL ---
    agent.eval()
    resource_visits = 0
    harm_events = 0
    total_steps = 0
    benefit_exposure_sum = 0.0
    # Track pre-arrival benefit scores: benefit_eval output BEFORE agent reaches a resource
    pre_arrival_benefit: List[float] = []

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_ttype = None
        prev_benefit_score = None

        for _ in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                    1, world_dim, device=agent.device
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks, temperature=0.5)

                # Current benefit_eval score (before action)
                z_world = latent.z_world
                curr_benefit_score = float(agent.e3.benefit_eval(z_world).item())

            _, reward, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            # Record pre-arrival benefit score (score just before a resource event)
            if ttype == "benefit_approach" and prev_benefit_score is not None:
                pre_arrival_benefit.append(prev_benefit_score)
                resource_visits += 1
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                harm_events += 1

            benefit_exposure_sum += float(obs_body[11] if obs_body.dim() == 1 else obs_body[0, 11])
            total_steps += 1
            prev_ttype = ttype
            prev_benefit_score = curr_benefit_score

            if done:
                break

    resource_visit_rate = resource_visits / max(1, total_steps)
    harm_rate = harm_events / max(1, total_steps)
    mean_benefit_exposure = benefit_exposure_sum / max(1, total_steps)
    mean_pre_arrival_benefit = sum(pre_arrival_benefit) / max(1, len(pre_arrival_benefit))

    print(
        f"  [eval] {cond_label} seed={seed}"
        f" resource_rate={resource_visit_rate:.4f}"
        f" harm_rate={harm_rate:.4f}"
        f" pre_arrival_benefit={mean_pre_arrival_benefit:.4f}"
        f" n_resource={resource_visits}"
        f" r2={benefit_eval_r2:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "benefit_eval_enabled": benefit_eval_enabled,
        "resource_visit_rate": resource_visit_rate,
        "harm_rate": harm_rate,
        "benefit_eval_r2": benefit_eval_r2,
        "mean_pre_arrival_benefit": mean_pre_arrival_benefit,
        "mean_benefit_exposure": mean_benefit_exposure,
        "n_resource_events": resource_visits,
        "n_harm_events": harm_events,
        "total_steps": total_steps,
    }


def run(
    seed: int = 42,
    warmup_episodes: int = 250,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    benefit_weight: float = 1.0,
    **kwargs,
) -> dict:
    """MECH-112: goal attractor -- NoGoOnly vs GoNoGo."""
    print(f"\n[EXQ-074] MECH-112 Goal Attractor", flush=True)

    r_nogo = _run_single(
        seed=seed, benefit_eval_enabled=False,
        warmup_episodes=warmup_episodes, eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        self_dim=self_dim, world_dim=world_dim, lr=lr, alpha_world=alpha_world,
    )
    r_gongo = _run_single(
        seed=seed, benefit_eval_enabled=True,
        warmup_episodes=warmup_episodes, eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        self_dim=self_dim, world_dim=world_dim, lr=lr, alpha_world=alpha_world,
        benefit_weight=benefit_weight,
    )

    rvr_nogo  = r_nogo["resource_visit_rate"]
    rvr_gongo = r_gongo["resource_visit_rate"]
    r2        = r_gongo["benefit_eval_r2"]
    pab_nogo  = r_nogo["mean_pre_arrival_benefit"]
    pab_gongo = r_gongo["mean_pre_arrival_benefit"]
    harm_nogo  = r_nogo["harm_rate"]
    harm_gongo = r_gongo["harm_rate"]

    c1_pass = rvr_gongo >= rvr_nogo + 0.05
    c2_pass = r2 > 0.10
    c3_pass = pab_gongo > pab_nogo + 0.02
    c4_pass = harm_gongo <= harm_nogo + 0.03

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[EXQ-074] Results:", flush=True)
    print(f"  resource_rate: nogo={rvr_nogo:.4f}  gongo={rvr_gongo:.4f}  gap={rvr_gongo-rvr_nogo:+.4f}", flush=True)
    print(f"  benefit_r2={r2:.4f}", flush=True)
    print(f"  pre_arrival_benefit: nogo={pab_nogo:.4f}  gongo={pab_gongo:.4f}  gap={pab_gongo-pab_nogo:+.4f}", flush=True)
    print(f"  harm_rate: nogo={harm_nogo:.4f}  gongo={harm_gongo:.4f}  delta={harm_gongo-harm_nogo:+.4f}", flush=True)
    print(f"  Status: {status} ({criteria_met}/4)", flush=True)

    if all_pass:
        interpretation = (
            "MECH-112 SUPPORTED: benefit_eval_head (prospective goal drive) increases"
            " resource-visit rate, head learns resource proximity (R2>0.10), and"
            " selected trajectories show higher pre-arrival benefit scores. The Go"
            " channel implements 'wanting' (prospective) distinct from 'liking'"
            " (reactive harm avoidance). Consistent with Barch & Dowd 2010."
        )
    elif criteria_met >= 2:
        interpretation = (
            "MECH-112 PARTIAL: Some goal-directed signal present but below threshold."
            " Longer training or stronger benefit signal may be needed."
        )
    else:
        interpretation = (
            "MECH-112 NOT SUPPORTED: benefit_eval_head does not produce measurable"
            " increase in resource-directed behaviour. Head may not be learning,"
            " benefit_weight may be too low relative to harm cost, or benefit_exposure"
            " signal is too weak for supervision."
        )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(f"C1 FAIL: resource gap={rvr_gongo-rvr_nogo:.4f} < 0.05")
    if not c2_pass:
        failure_notes.append(f"C2 FAIL: benefit_eval_r2={r2:.4f} <= 0.10")
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: pre_arrival gap={pab_gongo-pab_nogo:.4f} <= 0.02")
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: harm delta={harm_gongo-harm_nogo:.4f} > 0.03")
    for n in failure_notes:
        print(f"  {n}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = (
        f"# V3-EXQ-074 -- MECH-112 Goal Attractor (Benefit Eval Head)\n\n"
        f"**Status:** {status}\n**Claims:** MECH-112\n"
        f"**Seed:** {seed}  **Warmup:** {warmup_episodes}  **Eval:** {eval_episodes}\n"
        f"**benefit_weight:** {benefit_weight}\n\n"
        f"## Results\n\n"
        f"| Metric | NoGoOnly | GoNoGo | Delta |\n|---|---|---|---|\n"
        f"| resource_visit_rate | {rvr_nogo:.4f} | {rvr_gongo:.4f} | {rvr_gongo-rvr_nogo:+.4f} |\n"
        f"| benefit_eval_r2 | -- | {r2:.4f} | -- |\n"
        f"| pre_arrival_benefit | {pab_nogo:.4f} | {pab_gongo:.4f} | {pab_gongo-pab_nogo:+.4f} |\n"
        f"| harm_rate | {harm_nogo:.4f} | {harm_gongo:.4f} | {harm_gongo-harm_nogo:+.4f} |\n\n"
        f"## PASS Criteria\n\n| Criterion | Result |\n|---|---|\n"
        f"| C1: resource gap >= 0.05 | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: benefit_eval_r2 > 0.10 | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| C3: pre_arrival gap > 0.02 | {'PASS' if c3_pass else 'FAIL'} |\n"
        f"| C4: harm delta <= 0.03 | {'PASS' if c4_pass else 'FAIL'} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n{interpretation}\n{failure_section}\n"
    )

    metrics = {
        "resource_visit_rate_nogo":      float(rvr_nogo),
        "resource_visit_rate_gongo":     float(rvr_gongo),
        "resource_rate_gap":             float(rvr_gongo - rvr_nogo),
        "benefit_eval_r2":               float(r2),
        "pre_arrival_benefit_nogo":      float(pab_nogo),
        "pre_arrival_benefit_gongo":     float(pab_gongo),
        "pre_arrival_gap":               float(pab_gongo - pab_nogo),
        "harm_rate_nogo":                float(harm_nogo),
        "harm_rate_gongo":               float(harm_gongo),
        "harm_delta":                    float(harm_gongo - harm_nogo),
        "n_resource_events_nogo":        float(r_nogo["n_resource_events"]),
        "n_resource_events_gongo":       float(r_gongo["n_resource_events"]),
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
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--warmup",         type=int,   default=250)
    parser.add_argument("--eval-eps",       type=int,   default=50)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    parser.add_argument("--benefit-weight", type=float, default=1.0)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        benefit_weight=args.benefit_weight,
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
