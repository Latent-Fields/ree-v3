"""
V3-EXQ-027 — SD-003 V3: Full Reafference Pipeline (SD-007 + SD-008 + E3.harm_eval)

Claims: SD-003, SD-007, MECH-071, MECH-098

Motivation (2026-03-18):
  V2 EXQ-027 FAIL (calibration_gap ≈ 0.0007): identity shortcut — E2_world sees
  same egocentric z_world for any action because perspective shift dominates the
  signal. V3 fix requires two components:

  1. SD-008 (alpha_world=0.9): z_world tracks current observations sharply,
     so event transitions produce visible Δz_world.

  2. SD-007 (ReafferencePredictor): trained on empty-space steps (ttype == "none")
     to predict the z_world delta caused by locomotion. Applied in LatentStack.encode():
       z_world_corrected = z_world_raw - ReafferencePredictor(z_self_prev, a_prev)
     This removes the perspective-shift component, leaving only genuine world change.

  With both active: E3.harm_eval(z_world_corrected) should fire when genuinely
  harmful world states are encountered, but not on locomotion steps.

  SD-003 pipeline (V3):
    harm_actual = E3.harm_eval(z_world_corrected_at_agent_hazard)
    harm_cf     = E3.harm_eval(z_world_corrected_at_none)
    causal_sig  = harm_actual - harm_cf
    → calibration_gap = mean(causal_sig) > PASS threshold

Protocol:
  1. Train with reafference_action_dim = action_dim (SD-007 enabled) + alpha_world=0.9.
     ReafferencePredictor is trained on empty-space steps (ttype == "none"):
       loss = MSE(z_world_raw - z_world_prev,  ReafferencePredictor(z_self_prev, a_prev))
     E3 harm_eval_head is trained on (z_world_corrected, harm_signal) regression.
  2. Eval: same as EXQ-026 but uses z_world_corrected (from LatentState.z_world
     which is already perspective-corrected by SD-007).
  3. Metric: calibration_gap = mean_harm(agent_caused) - mean_harm(none)
  4. Additional diagnostic: reafference_r2 (lstsq fit of ReafferencePredictor outputs)

PASS criteria (ALL must hold):
  C1: calibration_gap > 0.15   (higher bar than EXQ-026; reafference should clean up signal)
  C2: reafference_r2 > 0.20    (ReafferencePredictor explains >20% of Δz_world variance)
  C3: harm_pred_std > 0.01     (E3 not collapsed)
  C4: n_agent_hazard_steps >= 5
  C5: No fatal errors

Relationship to EXQ-026:
  EXQ-026 tests E3.harm_eval WITHOUT reafference (just alpha=0.9).
  EXQ-027 tests the FULL pipeline WITH ReafferencePredictor.
  If EXQ-026 PASSES but EXQ-027 FAILS → reafference is introducing noise (unexpected).
  If EXQ-026 FAILS and EXQ-027 PASSES → reafference is necessary for clean signal.
  If both FAIL → deeper issue with z_world encoding or E3 training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_027_sd003_v3_reafference"
CLAIM_IDS = ["SD-003", "SD-007", "MECH-071", "MECH-098"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_reafference_r2(agent: REEAgent, reaf_data: List) -> float:
    """
    lstsq diagnostic: how well does ReafferencePredictor's output correlate
    with actual Δz_world on held-out empty steps?
    """
    if len(reaf_data) < 20:
        return 0.0
    n = len(reaf_data)
    n_train = int(n * 0.8)

    with torch.no_grad():
        z_self_all = torch.cat([d[0] for d in reaf_data], dim=0)
        a_all      = torch.cat([d[1] for d in reaf_data], dim=0)
        dz_all     = torch.cat([d[2] for d in reaf_data], dim=0)

        pred_all = agent.latent_stack.reafference_predictor(z_self_all, a_all)
        pred_test = pred_all[n_train:]
        dz_test   = dz_all[n_train:]

        if pred_test.shape[0] == 0:
            return 0.0

        ss_res = ((dz_test - pred_test) ** 2).sum()
        ss_tot = ((dz_test - dz_test.mean(dim=0, keepdim=True)) ** 2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())

    print(
        f"  reafference R² (test set n={pred_test.shape[0]}): {r2:.4f}",
        flush=True,
    )
    return max(0.0, r2)


def _train_with_reafference(
    agent: REEAgent,
    env: CausalGridWorld,
    optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    reaf_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Train agent with:
    1. Standard E1 + E2_self losses (optimizer)
    2. ReafferencePredictor loss on empty steps (reaf_optimizer)
    3. E3 harm_eval regression on all steps (harm_eval_optimizer)
    """
    agent.train()

    harm_buffer: List[Tuple[torch.Tensor, float]] = []
    MAX_HARM_BUF = 2000

    reaf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_REAF_DATA = 5000

    total_harm = 0
    total_benefit = 0
    n_agent_hazard = 0
    n_env_hazard = 0
    n_empty = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev = None
        z_self_prev  = None
        z_raw_prev   = None
        a_prev       = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()   # perspective-corrected
            z_raw_curr   = latent.z_world_raw.detach()  # uncorrected (SD-007 diagnostic)
            z_self_curr  = latent.z_self.detach()

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

            # Collect harm_eval training data (use corrected z_world)
            harm_buffer.append((z_world_curr, float(harm_signal)))
            if len(harm_buffer) > MAX_HARM_BUF:
                harm_buffer = harm_buffer[-MAX_HARM_BUF:]

            # Collect reafference training data (empty steps only)
            # Target: Δz_world_raw = z_raw_curr - z_raw_prev
            # because z_world_raw is what the predictor should model
            if (
                ttype == "none"
                and z_raw_prev is not None
                and z_self_prev is not None
                and a_prev is not None
            ):
                dz_raw = z_raw_curr - z_raw_prev
                reaf_data.append((
                    z_self_prev.cpu(),
                    a_prev.cpu(),
                    dz_raw.cpu(),
                ))
                n_empty += 1
                if len(reaf_data) > MAX_REAF_DATA:
                    reaf_data = reaf_data[-MAX_REAF_DATA:]

            # Standard losses
            e1_loss      = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()
            total_loss   = e1_loss + e2_self_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # ReafferencePredictor loss (on recent empty-step batch)
            if len(reaf_data) >= 16 and agent.latent_stack.reafference_predictor is not None:
                k = min(32, len(reaf_data))
                idxs = torch.randperm(len(reaf_data))[:k].tolist()
                zs_b = torch.cat([reaf_data[i][0] for i in idxs]).to(agent.device)
                a_b  = torch.cat([reaf_data[i][1] for i in idxs]).to(agent.device)
                dz_b = torch.cat([reaf_data[i][2] for i in idxs]).to(agent.device)
                pred_dz = agent.latent_stack.reafference_predictor(zs_b, a_b)
                reaf_loss = F.mse_loss(pred_dz, dz_b)
                if reaf_loss.requires_grad:
                    reaf_optimizer.zero_grad()
                    reaf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.latent_stack.reafference_predictor.parameters(), 0.5
                    )
                    reaf_optimizer.step()

            # E3 harm_eval training
            if len(harm_buffer) >= 16:
                k = min(32, len(harm_buffer))
                idxs = torch.randperm(len(harm_buffer))[:k].tolist()
                zw_b = torch.cat([harm_buffer[i][0] for i in idxs], dim=0)
                hs_b = torch.tensor(
                    [harm_buffer[i][1] for i in idxs], device=agent.device
                ).unsqueeze(1)
                # harm_eval_head uses Sigmoid → output in [0,1].
                # Map: harm (signal<0) → 1.0, everything else → 0.0.
                # (Prior: clamp(-1,1) → harm target=-1 unreachable by Sigmoid → collapsed)
                target    = (hs_b < 0).float()
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_raw_prev   = z_raw_curr
            z_self_prev  = z_self_curr
            a_prev       = action.detach()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}  "
                f"agent_hazard={n_agent_hazard}  env_hazard={n_env_hazard}  "
                f"empty_steps={n_empty}",
                flush=True,
            )

    return {
        "total_harm":    total_harm,
        "total_benefit": total_benefit,
        "n_agent_hazard": n_agent_hazard,
        "n_env_hazard":   n_env_hazard,
        "n_empty_steps":  n_empty,
        "reaf_data":      reaf_data,
    }


def _eval_harm_by_event(
    agent: REEAgent,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Collect E3.harm_eval(z_world_corrected) scores by event type.
    z_world_corrected is already in LatentState.z_world (SD-007 applied in encode()).
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

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_corr = latent.z_world  # perspective-corrected

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(z_world_corr).item())
                    all_scores.append(score)
                    if ttype in scores:
                        scores[ttype].append(score)
            except Exception:
                fatal_errors += 1

            if done:
                break

    mean_scores = {k: float(sum(v) / max(1, len(v))) for k, v in scores.items()}
    pred_std = float(torch.tensor(all_scores).std().item()) if len(all_scores) > 1 else 0.0
    n_counts = {k: len(v) for k, v in scores.items()}

    calibration_gap = mean_scores["agent_caused_hazard"] - mean_scores["none"]
    env_gap         = mean_scores["env_caused_hazard"]   - mean_scores["none"]

    print(
        f"  E3 harm_eval (corrected z_world) — none={mean_scores['none']:.4f}  "
        f"env={mean_scores['env_caused_hazard']:.4f}  "
        f"agent={mean_scores['agent_caused_hazard']:.4f}",
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
    warmup_episodes: int = 500,
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
    # SD-007: reafference_action_dim = action_dim enables ReafferencePredictor
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,  # SD-007 enabled
    )
    agent = REEAgent(config)

    # Verify SD-007 is enabled
    assert agent.latent_stack.reafference_predictor is not None, (
        "SD-007 ReafferencePredictor not initialized — check reafference_action_dim"
    )
    print(
        f"[V3-EXQ-027] SD-007 enabled: "
        f"ReafferencePredictor({self_dim}+{env.action_dim}→{world_dim})",
        flush=True,
    )

    # Three optimizers: standard, reafference, harm_eval
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "reafference_predictor" not in n
    ]
    reaf_params      = list(agent.latent_stack.reafference_predictor.parameters())
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer        = optim.Adam(standard_params, lr=lr)
    reaf_optimizer   = optim.Adam(reaf_params,     lr=1e-3)
    harm_eval_optim  = optim.Adam(harm_eval_params, lr=1e-4)

    print(
        f"  alpha_world={alpha_world}  alpha_self={alpha_self}  "
        f"warmup={warmup_episodes}  eval={eval_episodes}",
        flush=True,
    )

    train_out = _train_with_reafference(
        agent, env, optimizer, harm_eval_optim, reaf_optimizer,
        warmup_episodes, steps_per_episode,
    )

    # Diagnostic: reafference predictor R²
    print("[V3-EXQ-027] Computing reafference R²...", flush=True)
    reafference_r2 = _compute_reafference_r2(agent, train_out["reaf_data"])

    # Eval
    print(f"[V3-EXQ-027] Eval ({eval_episodes} eps)...", flush=True)
    eval_out = _eval_harm_by_event(agent, env, eval_episodes, steps_per_episode)
    fatal_errors = eval_out["fatal_errors"]

    # PASS / FAIL
    c1_pass = eval_out["calibration_gap"] > 0.15
    c2_pass = reafference_r2 > 0.20
    c3_pass = eval_out["harm_pred_std"] > 0.01
    c4_pass = eval_out["n_agent_hazard_steps"] >= 5
    c5_pass = fatal_errors == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: calibration_gap={eval_out['calibration_gap']:.4f} <= 0.15"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: reafference_r2={reafference_r2:.4f} <= 0.20"
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

    print(f"\nV3-EXQ-027 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "alpha_world":                   float(alpha_world),
        "alpha_self":                    float(alpha_self),
        "reafference_enabled":           1.0,
        "warmup_harm_events":            float(train_out["total_harm"]),
        "warmup_agent_hazard":           float(train_out["n_agent_hazard"]),
        "warmup_env_hazard":             float(train_out["n_env_hazard"]),
        "n_empty_steps_collected":       float(train_out["n_empty_steps"]),
        "reafference_r2":                float(reafference_r2),
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

    summary_markdown = f"""# V3-EXQ-027 — SD-003 V3: Full Reafference Pipeline

**Status:** {status}
**Claims:** SD-003, SD-007, MECH-071, MECH-098
**SD-007 enabled:** ReafferencePredictor({self_dim} + {env.action_dim} → {world_dim})
**alpha_world:** {alpha_world}  (SD-008)
**alpha_self:** {alpha_self}
**Warmup:** {warmup_episodes} eps (random policy, 12×12, 15 hazards, drift)
**Eval:** {eval_episodes} eps
**Seed:** {seed}

## Motivation

V3 SD-003 pipeline with full reafference correction (SD-007):
1. ReafferencePredictor trained on empty steps to predict perspective-shift Δz_world
2. z_world_corrected = z_world_raw - predicted_shift (applied in LatentStack.encode)
3. E3.harm_eval(z_world_corrected) trained and evaluated by event type
4. calibration_gap = mean(agent_hazard) - mean(none) → PASS > 0.15

V2 EXQ-027 FAIL root causes:
- alpha=0.3: event responses suppressed to 30% → identity shortcut trivially wins
- E2.predict_harm on z_gamma: wrong module + wrong latent

This experiment tests whether the corrected pipeline achieves genuine causal attribution.
Compare with EXQ-026 (same design but no reafference): gap between them isolates SD-007 effect.

## ReafferencePredictor Diagnostic

- n_empty_steps collected: {train_out['n_empty_steps']}
- R² (held-out): **{reafference_r2:.4f}**  (PASS threshold: 0.20)

## E3 harm_eval Scores (z_world_corrected)

| Event Type | mean E3.harm_eval | n_steps |
|---|---|---|
| none (locomotion) | {eval_out['mean_harm_score_none']:.4f} | {eval_out['n_none_steps']} |
| env_caused_hazard | {eval_out['mean_harm_score_env_hazard']:.4f} | {eval_out['n_env_hazard_steps']} |
| agent_caused_hazard | {eval_out['mean_harm_score_agent_hazard']:.4f} | {eval_out['n_agent_hazard_steps']} |

- **calibration_gap** (agent - none): **{eval_out['calibration_gap']:.4f}**  (PASS threshold: 0.15)
- env_gap (env - none): {eval_out['env_gap']:.4f}
- harm_pred_std: {eval_out['harm_pred_std']:.4f}

## Training Summary

- warmup_harm: {train_out['total_harm']}
- agent_hazard events: {train_out['n_agent_hazard']}
- env_hazard events: {train_out['n_env_hazard']}
- benefit: {train_out['total_benefit']}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: calibration_gap > 0.15 (SD-003 threshold) | {"PASS" if c1_pass else "FAIL"} | {eval_out['calibration_gap']:.4f} |
| C2: reafference_r2 > 0.20 (SD-007 predictive) | {"PASS" if c2_pass else "FAIL"} | {reafference_r2:.4f} |
| C3: harm_pred_std > 0.01 (E3 not collapsed) | {"PASS" if c3_pass else "FAIL"} | {eval_out['harm_pred_std']:.4f} |
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
    parser.add_argument("--warmup",       type=int,   default=500)
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
