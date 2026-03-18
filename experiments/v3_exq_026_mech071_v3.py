"""
V3-EXQ-026 — MECH-071 V3: E3.harm_eval Calibration Gap (alpha_world=0.9)

Claims: MECH-071, SD-003, SD-005, E3 (harm_eval)

Motivation (2026-03-18):
  MECH-071 (V2) used E2.predict_harm(z_gamma, a) to measure causal attribution.
  V3 redesign (per sd_004_sd_005_encoder_codesign.md §7): harm evaluation belongs
  on E3. The correct V3 attribution probe is:

      harm_actual = E3.harm_eval(z_world_after_agent_action)
      harm_env    = E3.harm_eval(z_world_after_env_action)
      gap         = mean(harm_actual) - mean(harm_env)

  This tests whether E3's harm_eval head, trained with alpha_world=0.9, can
  distinguish world states where the agent caused harm vs where the environment
  caused harm. No reafference correction is required here — the test is whether
  z_world reflects the harm-relevant difference at all.

  Prerequisite SD-008: alpha_world=0.9 so z_world tracks events sharply.
  Without alpha fix, z_world is a ~3-step EMA and E3 sees the same near-zero
  signal regardless of event type.

  This is a SIMPLER test than SD-003 counterfactual attribution. It does NOT
  require a counterfactual action. It just checks whether E3 can distinguish
  harm world-states from non-harm world-states by event origin.

Protocol:
  1. Train agent for warmup_episodes with alpha_world=0.9.
     E3 harm_eval_head is trained on (z_world, harm_signal) regression.
  2. Eval: collect z_world at steps where transition_type is:
     - "agent_caused_hazard": agent stepped into hazard
     - "env_caused_hazard": hazard moved into agent / env event
     - "none": empty locomotion steps
  3. Compute E3.harm_eval(z_world) for each group.
  4. calibration_gap = mean(agent_caused) - mean(none)
     env_gap          = mean(env_caused)  - mean(none)

PASS criteria (ALL must hold):
  C1: calibration_gap > 0.05  (agent-caused hazard states get higher harm score)
  C2: env_gap > 0.01           (env-caused hazard states also elevated — MECH-071
                                 env_caused contamination should also register)
  C3: harm_pred_std > 0.01    (E3 is not collapsed to constant prediction)
  C4: n_agent_hazard_steps >= 5  (enough events for reliable mean)
  C5: No fatal errors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_026_mech071_v3"
CLAIM_IDS = ["MECH-071", "SD-003", "SD-005"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _train_with_harm_eval(
    agent: REEAgent,
    env: CausalGridWorld,
    optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Train agent + E3 harm_eval head.
    E3 harm_eval is trained on (z_world, harm_signal) regression each step.
    """
    agent.train()

    # Balanced harm_eval training buffers.
    # Class imbalance fix: store harm and no-harm z_worlds separately.
    # With ~1.25% harm rate, an unbalanced buffer (all steps) collapses the
    # head to predict base-rate (~0.014) for everything.
    # Balanced sampling: 16 pos + 16 neg per batch → equal gradient signal.
    harm_buf_pos: List[torch.Tensor] = []  # z_world at harm steps (signal<0)
    harm_buf_neg: List[torch.Tensor] = []  # z_world at no-harm steps
    MAX_BUF_EACH = 1000

    total_harm = 0
    total_benefit = 0
    n_agent_hazard = 0
    n_env_hazard = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if harm_signal < 0:
                total_harm += 1
                if ttype == "agent_caused_hazard":
                    n_agent_hazard += 1
                elif ttype == "env_caused_hazard":
                    n_env_hazard += 1
            elif harm_signal > 0:
                total_benefit += 1

            # Collect for harm_eval training (balanced pos/neg buffers)
            zw = latent.z_world.detach()
            if harm_signal < 0:
                harm_buf_pos.append(zw)
                if len(harm_buf_pos) > MAX_BUF_EACH:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF_EACH:]
            else:
                harm_buf_neg.append(zw)
                if len(harm_buf_neg) > MAX_BUF_EACH:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF_EACH:]

            # Standard losses: E1 + E2_self
            e1_loss      = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()
            total_loss   = e1_loss + e2_self_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E3 harm_eval training: balanced 16 pos + 16 neg per batch.
            # harm_eval_head uses Sigmoid → output in [0,1].
            # target: harm step → 1.0, no-harm step → 0.0.
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b   = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)  # [batch, 1]
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}  "
                f"agent_hazard={n_agent_hazard}  env_hazard={n_env_hazard}",
                flush=True,
            )

    return {
        "total_harm":    total_harm,
        "total_benefit": total_benefit,
        "n_agent_hazard": n_agent_hazard,
        "n_env_hazard":   n_env_hazard,
    }


def _eval_harm_eval_by_event(
    agent: REEAgent,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Collect E3.harm_eval scores stratified by transition_type.

    Returns mean harm_eval score per event type.
    """
    agent.eval()
    scores: Dict[str, List[float]] = {
        "none": [],
        "env_caused_hazard": [],
        "agent_caused_hazard": [],
    }
    all_scores: List[float] = []
    fatal_errors = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        prev_ttype   = "none"

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_curr = latent.z_world  # [1, world_dim]

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            # Score z_world using E3.harm_eval
            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(z_world_curr).item())
                    all_scores.append(score)
                    if ttype in scores:
                        scores[ttype].append(score)
            except Exception:
                fatal_errors += 1

            z_world_prev = z_world_curr
            prev_ttype   = ttype
            if done:
                break

    mean_scores = {k: float(sum(v) / max(1, len(v))) for k, v in scores.items()}
    pred_std = float(torch.tensor(all_scores).std().item()) if len(all_scores) > 1 else 0.0
    n_counts = {k: len(v) for k, v in scores.items()}

    calibration_gap = mean_scores["agent_caused_hazard"] - mean_scores["none"]
    env_gap         = mean_scores["env_caused_hazard"]   - mean_scores["none"]

    print(
        f"  E3 harm_eval — none={mean_scores['none']:.4f}  "
        f"env_hazard={mean_scores['env_caused_hazard']:.4f}  "
        f"agent_hazard={mean_scores['agent_caused_hazard']:.4f}",
        flush=True,
    )
    print(
        f"  calibration_gap={calibration_gap:.4f}  env_gap={env_gap:.4f}  "
        f"pred_std={pred_std:.4f}  n={n_counts}",
        flush=True,
    )

    return {
        "mean_harm_score_none":         mean_scores["none"],
        "mean_harm_score_env_hazard":   mean_scores["env_caused_hazard"],
        "mean_harm_score_agent_hazard": mean_scores["agent_caused_hazard"],
        "calibration_gap":              calibration_gap,
        "env_gap":                      env_gap,
        "harm_pred_std":                pred_std,
        "n_none_steps":                 n_counts["none"],
        "n_env_hazard_steps":           n_counts["env_caused_hazard"],
        "n_agent_hazard_steps":         n_counts["agent_caused_hazard"],
        "fatal_errors":                 fatal_errors,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(
        seed=seed, size=12, num_hazards=15, num_resources=5,
        env_drift_interval=3, env_drift_prob=0.5,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
    )
    agent = REEAgent(config)

    # Separate optimizers: standard agent components + E3 harm_eval head
    agent_params = [p for n, p in agent.named_parameters()
                    if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer      = optim.Adam(agent_params,    lr=lr)
    harm_eval_optim = optim.Adam(harm_eval_params, lr=1e-4)

    print(
        f"[V3-EXQ-026] MECH-071 V3: E3.harm_eval calibration gap test",
        flush=True,
    )
    print(
        f"  alpha_world={alpha_world}  alpha_self={alpha_self}  "
        f"warmup={warmup_episodes}  eval={eval_episodes}",
        flush=True,
    )

    train_out = _train_with_harm_eval(
        agent, env, optimizer, harm_eval_optim,
        warmup_episodes, steps_per_episode,
    )

    print(f"[V3-EXQ-026] Eval ({eval_episodes} eps)...", flush=True)
    eval_out = _eval_harm_eval_by_event(
        agent, env, eval_episodes, steps_per_episode
    )
    fatal_errors = eval_out["fatal_errors"]

    # PASS / FAIL
    c1_pass = eval_out["calibration_gap"] > 0.05
    c2_pass = eval_out["env_gap"] > 0.01
    c3_pass = eval_out["harm_pred_std"] > 0.01
    c4_pass = eval_out["n_agent_hazard_steps"] >= 5
    c5_pass = fatal_errors == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: calibration_gap={eval_out['calibration_gap']:.4f} <= 0.05"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: env_gap={eval_out['env_gap']:.4f} <= 0.01"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: harm_pred_std={eval_out['harm_pred_std']:.4f} <= 0.01 (collapsed)"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_agent_hazard_steps={eval_out['n_agent_hazard_steps']} < 5"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={fatal_errors}")

    print(f"\nV3-EXQ-026 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "alpha_world":                   float(alpha_world),
        "alpha_self":                    float(alpha_self),
        "warmup_harm_events":            float(train_out["total_harm"]),
        "warmup_agent_hazard":           float(train_out["n_agent_hazard"]),
        "warmup_env_hazard":             float(train_out["n_env_hazard"]),
        "mean_harm_score_none":          float(eval_out["mean_harm_score_none"]),
        "mean_harm_score_env_hazard":    float(eval_out["mean_harm_score_env_hazard"]),
        "mean_harm_score_agent_hazard":  float(eval_out["mean_harm_score_agent_hazard"]),
        "calibration_gap":               float(eval_out["calibration_gap"]),
        "env_gap":                       float(eval_out["env_gap"]),
        "harm_pred_std":                 float(eval_out["harm_pred_std"]),
        "n_none_steps":                  float(eval_out["n_none_steps"]),
        "n_env_hazard_steps":            float(eval_out["n_env_hazard_steps"]),
        "n_agent_hazard_steps":          float(eval_out["n_agent_hazard_steps"]),
        "fatal_error_count":             float(fatal_errors),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-026 — MECH-071 V3: E3.harm_eval Calibration Gap

**Status:** {status}
**Claim:** MECH-071 V3 redesign (harm_eval belongs on E3, not E2)
**alpha_world:** {alpha_world}  (SD-008: must be >= 0.9 for event sensitivity)
**alpha_self:** {alpha_self}
**Warmup:** {warmup_episodes} eps (random policy, 12×12, 15 hazards, drift)
**Eval:** {eval_episodes} eps
**Seed:** {seed}

## Motivation

V3 redesign of MECH-071: harm evaluation lives on E3 (harm_eval_head), NOT on E2.
V2 approach (EXQ-027: E2.predict_harm calibration_gap ≈ 0.0007) failed because:
1. z_gamma (unified) dominated by perspective shift (SD-003 identity shortcut)
2. alpha=0.3 suppressed all event responses to ~30%

This test checks whether E3.harm_eval(z_world) can distinguish world states
corresponding to agent-caused hazard vs normal locomotion. Uses alpha_world={alpha_world}
so z_world responds sharply to events (SD-008 prerequisite).

## E3 harm_eval Scores by Event Type

| Event Type | mean E3.harm_eval | n_steps |
|---|---|---|
| none (locomotion) | {eval_out['mean_harm_score_none']:.4f} | {eval_out['n_none_steps']} |
| env_caused_hazard | {eval_out['mean_harm_score_env_hazard']:.4f} | {eval_out['n_env_hazard_steps']} |
| agent_caused_hazard | {eval_out['mean_harm_score_agent_hazard']:.4f} | {eval_out['n_agent_hazard_steps']} |

- **calibration_gap** (agent - none): **{eval_out['calibration_gap']:.4f}**
- **env_gap** (env - none): **{eval_out['env_gap']:.4f}**
- harm_pred_std: {eval_out['harm_pred_std']:.4f}

## Training Summary

- warmup_harm: {train_out['total_harm']}
- agent_hazard events: {train_out['n_agent_hazard']}
- env_hazard events: {train_out['n_env_hazard']}
- benefit: {train_out['total_benefit']}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: calibration_gap > 0.05 (agent-caused > none) | {"PASS" if c1_pass else "FAIL"} | {eval_out['calibration_gap']:.4f} |
| C2: env_gap > 0.01 (env-caused > none) | {"PASS" if c2_pass else "FAIL"} | {eval_out['env_gap']:.4f} |
| C3: harm_pred_std > 0.01 (not collapsed) | {"PASS" if c3_pass else "FAIL"} | {eval_out['harm_pred_std']:.4f} |
| C4: n_agent_hazard_steps >= 5 | {"PASS" if c4_pass else "FAIL"} | {eval_out['n_agent_hazard_steps']} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {fatal_errors} |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": fatal_errors,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--warmup",       type=int,   default=400)
    parser.add_argument("--eval-eps",     type=int,   default=50)
    parser.add_argument("--steps",        type=int,   default=200)
    parser.add_argument("--alpha-world",  type=float, default=0.9)
    parser.add_argument("--alpha-self",   type=float, default=0.3)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
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
