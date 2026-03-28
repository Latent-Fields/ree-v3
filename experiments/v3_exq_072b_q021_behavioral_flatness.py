"""
V3-EXQ-072b -- Q-021 Behavioral Flatness Diagnostic

Claims: Q-021
Supersedes: V3-EXQ-072 (crashed 2026-03-23 with exit code 1 in ~3 sec; no
  traceback preserved. All APIs used confirmed present as of 2026-03-28.
  Fix: use env.body_obs_dim / env.world_obs_dim instead of hardcoded dims.)

Q-021 asks whether a pure-avoidance architecture (NoGo-only, no Go channel)
produces behavioral flatness -- i.e., resource-visit rates <= random baseline
because the gradient minimum under harm-only training is quiescence.

Three conditions (matched seeds):
  A. NoGo-only  -- current architecture, benefit_eval_enabled=False
  B. Competitive -- benefit_eval_enabled=True, benefit_weight=1.0 (ARC-030 Go channel)
  C. Random     -- agent selects uniformly random actions (baseline floor)

The environment has symmetric approach/avoidance structure (CausalGridWorldV2
use_proxy_fields=True) so any asymmetry in resource-visiting is attributable
to the evaluation architecture, not the environment.

PASS criteria (confirming Q-021 behavioral flatness hypothesis):
  C1: resource_visit_rate_nogo <= resource_visit_rate_random + 0.02
      (NoGo-only no better than random at resource seeking)
  C2: resource_visit_rate_competitive >= resource_visit_rate_nogo + 0.05
      (Go channel enables systematic resource approach)
  C3: policy_entropy_nogo <= policy_entropy_competitive - 0.10
      (NoGo-only has lower action diversity = more avoidant/flat behaviour)
  C4: n_resource_events_nogo <= n_resource_events_competitive * 0.80
      (Fewer benefit_approach events in NoGo-only condition)

Informational metrics:
  harm_rate per condition (all should be similar -- harm avoidance should work)
  resource_field_at_agent (how close agent stays to resource gradient)

Biological basis:
  Bariselli 2018 (PMID 29481617): D1/D2 pathways evaluate SAME proposals.
  Without D1 (Go), only D2 (NoGo) constrains selection -- quiescent default.
  Barch & Dowd 2010 (PMID 20868638): "wanting" (prospective) vs "liking"
  (reactive). NoGo-only = liking without wanting. Flatness = avolition.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
from typing import Dict, List, Optional
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_072b_q021_behavioral_flatness"
CLAIM_IDS = ["Q-021"]

ACTION_DIM = 4


def _make_env(seed: int) -> CausalGridWorld:
    env = CausalGridWorld(
        size=10,
        num_resources=5,
        num_hazards=3,
        use_proxy_fields=True,
        seed=seed,
    )
    return env


def _action_to_onehot(action_idx: int, device) -> torch.Tensor:
    v = torch.zeros(1, ACTION_DIM, device=device)
    v[0, action_idx] = 1.0
    return v


def _action_entropy(action_counts: List[int]) -> float:
    """Shannon entropy of action distribution."""
    total = sum(action_counts) + 1e-8
    probs = [c / total for c in action_counts]
    return -sum(p * math.log(p + 1e-9) for p in probs if p > 0)


def _run_nogo_or_competitive(
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
    """Run NoGo-only or Competitive condition."""
    cond_label = "COMPETITIVE" if benefit_eval_enabled else "NOGO_ONLY"

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)

    # Use env dims directly -- avoids stale hardcoded constants
    body_obs_dim  = env.body_obs_dim
    world_obs_dim = env.world_obs_dim

    config = REEConfig.from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        benefit_eval_enabled=benefit_eval_enabled,
        benefit_weight=benefit_weight,
    )
    agent = REEAgent(config)

    # Separate optimisers for E1, E2, E3 (consistent with other V3 experiments)
    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)

    print(f"\n[EXQ-072b] TRAIN {cond_label} seed={seed}", flush=True)
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        total_harm = 0.0

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

            action_idx = int(action.squeeze().argmax().item())
            _, reward, done, info, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            total_harm += abs(harm_signal)

            # E1 + E2 training (combined backward)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_e1_e2 = e1_loss + e2_loss
            if total_e1_e2.requires_grad:
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total_e1_e2.backward()
                e1_opt.step()
                e2_opt.step()

            # E3 harm loss from actual harm signal
            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_pred = agent.e3.harm_eval(z_world)
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(harm_pred, harm_target)
                e3_loss = harm_loss

                # Benefit eval loss (when competitive)
                if benefit_eval_enabled:
                    benefit_exp = obs_body[11] if obs_body.dim() == 1 else obs_body[0, 11]
                    benefit_target = torch.tensor(
                        [[float(benefit_exp)]], device=agent.device
                    )
                    benefit_loss = agent.compute_benefit_eval_loss(benefit_target)
                    if benefit_loss.requires_grad:
                        e3_loss = e3_loss + benefit_loss

                e3_opt.zero_grad()
                e3_loss.backward()
                e3_opt.step()

            agent.update_residue(harm_signal)

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"  [train] {cond_label} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm={total_harm:.3f}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()
    resource_visits = 0
    harm_events = 0
    benefit_events = 0
    action_counts = [0] * ACTION_DIM
    total_steps = 0
    resource_field_sum = 0.0

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

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

            action_idx = int(action.squeeze().argmax().item())
            action_counts[action_idx] += 1

            _, reward, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype == "benefit_approach":
                resource_visits += 1
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                harm_events += 1

            resource_field_sum += float(info.get("resource_field_at_agent", 0.0))
            total_steps += 1

            if done:
                break

    resource_visit_rate = resource_visits / max(1, total_steps)
    harm_rate = harm_events / max(1, total_steps)
    policy_entropy = _action_entropy(action_counts)
    mean_resource_field = resource_field_sum / max(1, total_steps)

    print(
        f"  [eval] {cond_label} seed={seed}"
        f" resource_rate={resource_visit_rate:.4f}"
        f" harm_rate={harm_rate:.4f}"
        f" entropy={policy_entropy:.4f}"
        f" mean_rf={mean_resource_field:.4f}"
        f" n_resource={resource_visits}"
        f" total_steps={total_steps}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "benefit_eval_enabled": benefit_eval_enabled,
        "resource_visit_rate": resource_visit_rate,
        "harm_rate": harm_rate,
        "policy_entropy": policy_entropy,
        "mean_resource_field": mean_resource_field,
        "n_resource_events": resource_visits,
        "n_harm_events": harm_events,
        "total_steps": total_steps,
    }


def _run_random(
    seed: int,
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """Random baseline -- uniform action selection."""
    random.seed(seed)
    env = _make_env(seed)

    resource_visits = 0
    harm_events = 0
    action_counts = [0] * ACTION_DIM
    total_steps = 0
    resource_field_sum = 0.0

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()

        for _ in range(steps_per_episode):
            action_idx = random.randint(0, ACTION_DIM - 1)
            action = _action_to_onehot(action_idx, "cpu")
            action_counts[action_idx] += 1

            _, reward, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype == "benefit_approach":
                resource_visits += 1
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                harm_events += 1
            resource_field_sum += float(info.get("resource_field_at_agent", 0.0))
            total_steps += 1

            if done:
                break

    resource_visit_rate = resource_visits / max(1, total_steps)
    harm_rate = harm_events / max(1, total_steps)
    policy_entropy = _action_entropy(action_counts)
    mean_resource_field = resource_field_sum / max(1, total_steps)

    print(
        f"  [eval] RANDOM seed={seed}"
        f" resource_rate={resource_visit_rate:.4f}"
        f" harm_rate={harm_rate:.4f}"
        f" entropy={policy_entropy:.4f}"
        f" mean_rf={mean_resource_field:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": "RANDOM",
        "benefit_eval_enabled": False,
        "resource_visit_rate": resource_visit_rate,
        "harm_rate": harm_rate,
        "policy_entropy": policy_entropy,
        "mean_resource_field": mean_resource_field,
        "n_resource_events": resource_visits,
        "n_harm_events": harm_events,
        "total_steps": total_steps,
    }


def run(
    seed: int = 42,
    warmup_episodes: int = 200,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    """
    Q-021 behavioral flatness: NoGo-only vs Competitive vs Random.
    """
    print(f"\n[EXQ-072b] Q-021 Behavioral Flatness Diagnostic", flush=True)
    print(f"  seed={seed} warmup={warmup_episodes} eval={eval_episodes}", flush=True)

    r_nogo = _run_nogo_or_competitive(
        seed=seed,
        benefit_eval_enabled=False,
        warmup_episodes=warmup_episodes,
        eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        self_dim=self_dim,
        world_dim=world_dim,
        lr=lr,
        alpha_world=alpha_world,
    )

    r_comp = _run_nogo_or_competitive(
        seed=seed,
        benefit_eval_enabled=True,
        warmup_episodes=warmup_episodes,
        eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        self_dim=self_dim,
        world_dim=world_dim,
        lr=lr,
        alpha_world=alpha_world,
        benefit_weight=1.0,
    )

    r_rand = _run_random(
        seed=seed,
        eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
    )

    # PASS criteria
    rvr_nogo = r_nogo["resource_visit_rate"]
    rvr_comp = r_comp["resource_visit_rate"]
    rvr_rand = r_rand["resource_visit_rate"]
    ent_nogo = r_nogo["policy_entropy"]
    ent_comp = r_comp["policy_entropy"]
    nr_nogo  = r_nogo["n_resource_events"]
    nr_comp  = r_comp["n_resource_events"]

    c1_pass = rvr_nogo <= rvr_rand + 0.02
    c2_pass = rvr_comp >= rvr_nogo + 0.05
    c3_pass = ent_nogo <= ent_comp - 0.10
    c4_pass = nr_nogo <= nr_comp * 0.80

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[EXQ-072b] Results:", flush=True)
    print(f"  resource_visit_rate: nogo={rvr_nogo:.4f} comp={rvr_comp:.4f} rand={rvr_rand:.4f}", flush=True)
    print(f"  policy_entropy:      nogo={ent_nogo:.4f} comp={ent_comp:.4f}", flush=True)
    print(f"  n_resource_events:   nogo={nr_nogo} comp={nr_comp}", flush=True)
    print(f"  C1 nogo<=rand+0.02:  {'PASS' if c1_pass else 'FAIL'}", flush=True)
    print(f"  C2 comp>=nogo+0.05:  {'PASS' if c2_pass else 'FAIL'}", flush=True)
    print(f"  C3 entropy gap>=0.10:{'PASS' if c3_pass else 'FAIL'}", flush=True)
    print(f"  C4 nogo<=comp*0.80:  {'PASS' if c4_pass else 'FAIL'}", flush=True)
    print(f"  Status: {status} ({criteria_met}/4)", flush=True)

    if all_pass:
        interpretation = (
            "Q-021 CONFIRMED: Pure-avoidance (NoGo-only) architecture produces"
            " behavioral flatness. Resource-visit rate at or below random baseline;"
            " competitive Go/NoGo channel restores systematic approach. This"
            " supports ARC-030 (Go/NoGo competitive evaluation) and validates the"
            " design motivation for MECH-112 (benefit_eval goal attractor)."
        )
    elif criteria_met >= 2:
        interpretation = (
            "Q-021 PARTIAL: Some behavioral flatness signal detected but below"
            " threshold. Longer training or stronger harm signal may be needed."
        )
    else:
        interpretation = (
            "Q-021 NOT CONFIRMED: NoGo-only agent not demonstrably flatter than"
            " random baseline. Possible: harm signal too weak, training too short,"
            " or flatness hypothesis incorrect."
        )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: nogo resource_rate={rvr_nogo:.4f} > random+0.02={rvr_rand+0.02:.4f}"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: comp resource_rate={rvr_comp:.4f} < nogo+0.05={rvr_nogo+0.05:.4f}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: entropy gap={ent_comp - ent_nogo:.4f} < 0.10"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: nogo resource events={nr_nogo} > comp*0.80={nr_comp*0.80:.1f}"
        )
    for n in failure_notes:
        print(f"  {n}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = (
        f"# V3-EXQ-072b -- Q-021 Behavioral Flatness Diagnostic\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** Q-021\n"
        f"**Seed:** {seed}  **Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n\n"
        f"## Conditions\n\n"
        f"| Condition | resource_visit_rate | policy_entropy | n_resource_events |\n"
        f"|---|---|---|---|\n"
        f"| NoGo-only (A) | {rvr_nogo:.4f} | {ent_nogo:.4f} | {nr_nogo} |\n"
        f"| Competitive (B) | {rvr_comp:.4f} | {ent_comp:.4f} | {nr_comp} |\n"
        f"| Random baseline | {rvr_rand:.4f} | {r_rand['policy_entropy']:.4f} | {r_rand['n_resource_events']} |\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: nogo <= random+0.02 (flatness present) | {'PASS' if c1_pass else 'FAIL'} | {rvr_nogo:.4f} vs {rvr_rand+0.02:.4f} |\n"
        f"| C2: competitive >= nogo+0.05 (Go channel helps) | {'PASS' if c2_pass else 'FAIL'} | {rvr_comp:.4f} vs {rvr_nogo+0.05:.4f} |\n"
        f"| C3: entropy gap >= 0.10 (diversity difference) | {'PASS' if c3_pass else 'FAIL'} | {ent_comp-ent_nogo:.4f} |\n"
        f"| C4: nogo resource events <= comp*0.80 | {'PASS' if c4_pass else 'FAIL'} | {nr_nogo} vs {nr_comp*0.80:.1f} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n{interpretation}\n{failure_section}\n"
    )

    metrics = {
        "resource_visit_rate_nogo":  float(rvr_nogo),
        "resource_visit_rate_comp":  float(rvr_comp),
        "resource_visit_rate_rand":  float(rvr_rand),
        "policy_entropy_nogo":       float(ent_nogo),
        "policy_entropy_comp":       float(ent_comp),
        "n_resource_events_nogo":    float(nr_nogo),
        "n_resource_events_comp":    float(nr_comp),
        "harm_rate_nogo":            float(r_nogo["harm_rate"]),
        "harm_rate_comp":            float(r_comp["harm_rate"]),
        "mean_resource_field_nogo":  float(r_nogo["mean_resource_field"]),
        "mean_resource_field_comp":  float(r_comp["mean_resource_field"]),
        "crit1_pass":               1.0 if c1_pass else 0.0,
        "crit2_pass":               1.0 if c2_pass else 0.0,
        "crit3_pass":               1.0 if c3_pass else 0.0,
        "crit4_pass":               1.0 if c4_pass else 0.0,
        "criteria_met":             float(criteria_met),
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
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--warmup",      type=int,   default=200)
    parser.add_argument("--eval-eps",    type=int,   default=50)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
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
