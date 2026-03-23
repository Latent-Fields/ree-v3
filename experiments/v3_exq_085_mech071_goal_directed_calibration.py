#!/opt/local/bin/python3
"""
V3-EXQ-085 -- MECH-071 Goal-Directed Harm Calibration

Claims: MECH-071, INV-034

Prior MECH-071 experiments (EXQ-079) failed because n_agent_min=0: the agent
generated zero agent-caused harm events. Without agent-caused proximity
transitions, the calibration asymmetry claim cannot be tested.

Root cause: a pure harm-avoidance system (no positive goal) drifts to the
corners and stays there. The agent never purposefully approaches a hazard,
so "agent_caused_hazard" transitions never occur.

Fix: enable z_goal substrate. With a resource that the agent wants to reach,
the agent will navigate the environment -- including near hazards -- generating
the agent-caused proximity events needed to test MECH-071.

This is also a direct test of INV-034: an agent with only harm avoidance cannot
exercise genuine agency. The goal-directed condition should both generate more
agent-caused harm events AND show better E3 harm calibration.

Discriminative pair:
  GOAL_PRESENT  -- z_goal_enabled=True, benefit_eval enabled, e1_goal_conditioned=True
                   Agent has positive motivation to approach resources, navigates
                   environment, generates agent-caused approach events naturally.
  GOAL_ABSENT   -- z_goal_enabled=False, no benefit supervision (baseline).
                   Agent uses random actions only. Should show near-zero
                   agent-caused events (replicating EXQ-079 null result).

MECH-071 claim:
  After training with z_goal active, E3.harm_eval should show asymmetric
  calibration: HIGHER scores on agent-caused approach transitions than on
  env-caused or locomotion transitions. This asymmetry indicates E3 has
  learned to track the agent's own causal role in harm proximity.

INV-034 connection:
  The goal-present condition demonstrates that goal-directed motivation
  (wanting) is a *prerequisite* for generating the experience needed to
  calibrate harm attribution. Without a goal, there is no navigation,
  no exposure to agent-caused proximity, and therefore no data for E3 to
  learn the calibration asymmetry.

PASS criteria (ALL required):
  C1: n_agent_events_goal_present >= 15
      (goal generates agent-caused proximity events; resolves EXQ-079 null)
  C2: calibration_gap_agent_goal_present > 0.02
      (E3 distinguishes agent-caused approach from baseline none)
  C3: n_agent_events_goal_present > n_agent_events_goal_absent * 1.5
      (goal reliably increases agent-caused events over random baseline)
  C4: no fatal errors

Decision scoring:
  retain_ree:       C1+C2+C3 pass
  hybridize:        C1+C3 pass but C2 marginal (events present, calibration weak)
  retire_ree_claim: C1 fails (even with goal, no agent-caused events)
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


EXPERIMENT_TYPE = "v3_exq_085_mech071_goal_directed_calibration"
CLAIM_IDS = ["MECH-071", "INV-034"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_single(
    seed: int,
    goal_present: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
) -> Dict:
    """Run one (seed, condition) cell and return calibration + event count metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"

    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
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
        alpha_self=alpha_self,
        reafference_action_dim=0,
    )

    # Enable z_goal in goal-present condition
    if goal_present:
        config.goal.z_goal_enabled = True
        config.goal.e1_goal_conditioned = True
        config.goal.alpha_goal = 0.05
        config.goal.decay_goal = 0.005

    agent = REEAgent(config)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    benefit_buf: List[Tuple[torch.Tensor, float]] = []
    MAX_BUF = 2000

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    train_counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0,
    }

    # --- WARMUP TRAINING ---
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            # Goal-directed action selection: if z_goal active, use goal proximity
            # to bias action. Otherwise random.
            action_idx: int
            if goal_present and agent.goal_state.is_active():
                # Simple greedy goal approach: pick action that maximises goal proximity
                # among a random subset of actions (3-sample lookahead using agent memory)
                best_action = random.randint(0, env.action_dim - 1)
                best_prox = -1.0
                for a_try in range(env.action_dim):
                    a_tensor = _action_to_onehot(a_try, env.action_dim, agent.device)
                    with torch.no_grad():
                        z_world_pred = agent.e2.world_forward(z_world_curr, a_tensor)
                        prox = float(agent.goal_state.goal_proximity(z_world_pred))
                    if prox > best_prox:
                        best_prox = prox
                        best_action = a_try
                # Mix goal-directed (60%) with random (40%) to ensure exploration
                if random.random() < 0.6:
                    action_idx = best_action
                else:
                    action_idx = random.randint(0, env.action_dim - 1)
            else:
                action_idx = random.randint(0, env.action_dim - 1)

            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            benefit_signal = info.get("benefit_signal", 0.0)

            if ttype in train_counts:
                train_counts[ttype] += 1

            # Update goal from benefit signal
            if goal_present and benefit_signal > 0:
                agent.update_z_goal(float(benefit_signal))

            # Harm buffer
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E3 harm_eval training
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            agent_events = (
                train_counts["agent_caused_hazard"] + train_counts["hazard_approach"]
            )
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" agent_events={agent_events}"
                f" env_caused={train_counts['env_caused_hazard']}"
                f" none={train_counts['none']}",
                flush=True,
            )

    # --- EVAL: measure harm calibration by transition type ---
    agent.eval()

    scores: Dict[str, List[float]] = {
        "none": [], "hazard_approach": [],
        "env_caused_hazard": [], "agent_caused_hazard": [],
    }
    n_fatal = 0

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_curr = latent.z_world

            if goal_present and agent.goal_state.is_active():
                best_action = random.randint(0, env.action_dim - 1)
                best_prox = -1.0
                for a_try in range(env.action_dim):
                    a_tensor = _action_to_onehot(a_try, env.action_dim, agent.device)
                    with torch.no_grad():
                        z_world_pred = agent.e2.world_forward(z_world_curr, a_tensor)
                        prox = float(agent.goal_state.goal_proximity(z_world_pred))
                    if prox > best_prox:
                        best_prox = prox
                        best_action = a_try
                action_idx = best_action if random.random() < 0.6 else random.randint(
                    0, env.action_dim - 1,
                )
            else:
                action_idx = random.randint(0, env.action_dim - 1)

            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            benefit_signal = info.get("benefit_signal", 0.0)

            if goal_present and benefit_signal > 0:
                agent.update_z_goal(float(benefit_signal))

            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(z_world_curr).item())
                if ttype in scores:
                    scores[ttype].append(score)
            except Exception:
                n_fatal += 1

            if done:
                break

    means: Dict[str, float] = {
        k: float(sum(v) / max(1, len(v))) for k, v in scores.items()
    }
    n_counts = {k: len(v) for k, v in scores.items()}

    n_agent_events = (
        n_counts.get("agent_caused_hazard", 0) + n_counts.get("hazard_approach", 0)
    )
    # Calibration gap: agent-caused vs none
    mean_agent = (
        (means["agent_caused_hazard"] + means["hazard_approach"]) / 2.0
        if (n_counts.get("agent_caused_hazard", 0) > 0 or n_counts.get("hazard_approach", 0) > 0)
        else means["hazard_approach"]
    )
    calibration_gap_agent = mean_agent - means["none"]
    calibration_gap_env   = means["env_caused_hazard"] - means["none"]

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" none={means['none']:.4f}"
        f" agent_approach={means['hazard_approach']:.4f}"
        f" agent_caused={means['agent_caused_hazard']:.4f}"
        f" env_caused={means['env_caused_hazard']:.4f}"
        f" n_agent={n_agent_events}"
        f" cal_gap_agent={calibration_gap_agent:.4f}"
        f" cal_gap_env={calibration_gap_env:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "goal_present": goal_present,
        "n_agent_events": int(n_agent_events),
        "n_env_events": int(n_counts.get("env_caused_hazard", 0)),
        "n_none": int(n_counts.get("none", 0)),
        "calibration_gap_agent": float(calibration_gap_agent),
        "calibration_gap_env":   float(calibration_gap_env),
        "mean_score_none":       float(means["none"]),
        "mean_score_agent":      float(mean_agent),
        "mean_score_env":        float(means["env_caused_hazard"]),
        "train_agent_events": int(
            train_counts["agent_caused_hazard"] + train_counts["hazard_approach"]
        ),
        "n_fatal": int(n_fatal),
    }


def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 300,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    **kwargs,
) -> dict:
    """Discriminative pair: GOAL_PRESENT vs GOAL_ABSENT for MECH-071 calibration."""
    results_goal: List[Dict] = []
    results_base: List[Dict] = []

    for seed in seeds:
        for goal_present in [True, False]:
            label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"
            print(
                f"\n[V3-EXQ-085] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                goal_present=goal_present,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
            )
            if goal_present:
                results_goal.append(r)
            else:
                results_base.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    n_agent_goal   = _avg(results_goal, "n_agent_events")
    n_agent_base   = _avg(results_base, "n_agent_events")
    cal_gap_agent  = _avg(results_goal, "calibration_gap_agent")
    cal_gap_base   = _avg(results_base, "calibration_gap_agent")

    # Pre-registered PASS criteria
    c1_pass = n_agent_goal   >= 15
    c2_pass = cal_gap_agent  > 0.02
    c3_pass = n_agent_goal   > n_agent_base * 1.5
    c4_pass = all(r["n_fatal"] == 0 for r in results_goal + results_base)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c3_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-085] Final results:", flush=True)
    print(
        f"  n_agent_events: goal={n_agent_goal:.1f}  base={n_agent_base:.1f}",
        flush=True,
    )
    print(
        f"  calibration_gap: goal={cal_gap_agent:.4f}  base={cal_gap_base:.4f}",
        flush=True,
    )
    print(f"  decision={decision}  status={status} ({criteria_met}/4)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: n_agent_events_goal={n_agent_goal:.1f} < 15"
            " (goal not generating agent-caused proximity events)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: cal_gap_agent={cal_gap_agent:.4f} <= 0.02"
            " (E3 not calibrated for agent-caused approach)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: goal={n_agent_goal:.1f} not > base={n_agent_base:.1f} * 1.5"
            " (goal condition not reliably increasing agent events)"
        )
    if not c4_pass:
        failure_notes.append("C4 FAIL: fatal errors in one or more runs")

    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "n_agent_events_goal_present": float(n_agent_goal),
        "n_agent_events_goal_absent":  float(n_agent_base),
        "calibration_gap_goal_present": float(cal_gap_agent),
        "calibration_gap_goal_absent":  float(cal_gap_base),
        "delta_n_agent_events":         float(n_agent_goal - n_agent_base),
        "delta_calibration_gap":        float(cal_gap_agent - cal_gap_base),
        "n_seeds":                      float(len(seeds)),
        "alpha_world":                  float(alpha_world),
        "crit1_pass":                   1.0 if c1_pass else 0.0,
        "crit2_pass":                   1.0 if c2_pass else 0.0,
        "crit3_pass":                   1.0 if c3_pass else 0.0,
        "crit4_pass":                   1.0 if c4_pass else 0.0,
        "criteria_met":                 float(criteria_met),
    }

    per_goal_rows = "\n".join(
        f"  seed={r['seed']}: n_agent={r['n_agent_events']}"
        f" cal_gap={r['calibration_gap_agent']:.4f}"
        f" train_agent={r['train_agent_events']}"
        for r in results_goal
    )
    per_base_rows = "\n".join(
        f"  seed={r['seed']}: n_agent={r['n_agent_events']}"
        f" cal_gap={r['calibration_gap_agent']:.4f}"
        for r in results_base
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-085 -- MECH-071 Goal-Directed Harm Calibration\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-071, INV-034\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**Design note:** EXQ-079 redesign -- z_goal substrate enables agent-caused events.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: n_agent_events_goal_present >= 15\n"
        f"C2: calibration_gap_goal_present > 0.02\n"
        f"C3: n_agent_events_goal_present > n_agent_events_goal_absent * 1.5\n"
        f"C4: no fatal errors\n\n"
        f"## Results\n\n"
        f"| Condition | n_agent_events | calibration_gap_agent |\n"
        f"|-----------|---------------|----------------------|\n"
        f"| GOAL_PRESENT | {n_agent_goal:.1f} | {cal_gap_agent:.4f} |\n"
        f"| GOAL_ABSENT  | {n_agent_base:.1f} | {cal_gap_base:.4f} |\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: n_agent_goal >= 15 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {n_agent_goal:.1f} |\n"
        f"| C2: cal_gap_agent > 0.02 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {cal_gap_agent:.4f} |\n"
        f"| C3: goal > base * 1.5 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {n_agent_goal:.1f} vs {n_agent_base:.1f} |\n"
        f"| C4: no fatal errors | {'PASS' if c4_pass else 'FAIL'} | -- |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Per-Seed\n\n"
        f"GOAL_PRESENT:\n{per_goal_rows}\n\n"
        f"GOAL_ABSENT:\n{per_base_rows}\n"
        f"{failure_section}\n"
    )

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
        "fatal_error_count": sum(
            r["n_fatal"] for r in results_goal + results_base
        ),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 7])
    parser.add_argument("--warmup",          type=int,   default=300)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
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
