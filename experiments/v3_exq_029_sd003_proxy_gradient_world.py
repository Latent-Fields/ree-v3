"""
V3-EXQ-029 — SD-003 on CausalGridWorldV2: Proxy-Gradient World + Full Pipeline

Claims: SD-003, SD-007, ARC-024, MECH-071

This is EXQ-027 adapted for CausalGridWorldV2 (use_proxy_fields=True).

The key question: does a world that generates observable harm/benefit gradient signals
*before* contact events enable E3.harm_eval to detect hazard approach — not just
hazard contact?

EXQ-027 on original world (contact-only harm):
  - harm signals only at contact → sparse training signal for E3
  - E2.world_forward sees same z_world for "move toward hazard" and "move away"
  - calibration_gap consistently near 0.0 (E3 has no gradient to learn)

EXQ-029 on CausalGridWorldV2 (proxy-gradient harm):
  - harm signals at approach AND contact → dense gradient training signal for E3
  - E2.world_forward sees different z_world for different distances from hazard
  - hazard_field_view in world_obs gives direct gradient information
  - HYPOTHESIS: calibration_gap_approach = E3(hazard_approach) - E3(none) > 0.08

Architecture basis (ARC-024):
  Harm/benefit signals are proxies along gradients toward asymptotic limits
  (death / love). A world generating gradient fields makes these proxies
  observable before their limit is reached. This is the environmental fix
  required by the EXQ-006/EXQ-020/EXQ-027 failure cluster.

Protocol:
  1. CausalGridWorldV2: use_proxy_fields=True, proximity_harm_scale=0.05
  2. SD-007 enabled: ReafferencePredictor(z_world_raw_prev, a_prev) — MECH-101
  3. SD-008: alpha_world=0.9
  4. Train: E3.harm_eval on (z_world_corrected, harm_signal) regression.
     harm_buf_pos includes BOTH hazard_approach AND contact steps.
  5. Eval: measure E3.harm_eval by transition_type including "hazard_approach"
  6. Key metric: calibration_gap_approach = E3(hazard_approach) - E3(none)

PASS criteria (ALL must hold):
  C1: calibration_gap_approach > 0.08  (approach detectable, lower than contact-only 0.15)
  C2: calibration_gap_contact  > 0.05  (contact still detected after gradient training)
  C3: reafference_r2 > 0.10            (ReafferencePredictor reduces perspective noise)
  C4: harm_pred_std > 0.01             (E3 not collapsed)
  C5: n_hazard_approach_steps >= 30    (sufficient approach events)
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


EXPERIMENT_TYPE = "v3_exq_029_sd003_proxy_gradient_world"
CLAIM_IDS = ["SD-003", "SD-007", "ARC-024", "MECH-071"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_reafference_r2(agent: REEAgent, reaf_data: List) -> float:
    if len(reaf_data) < 20:
        return 0.0
    n = len(reaf_data)
    n_train = int(n * 0.8)

    with torch.no_grad():
        z_world_raw_all = torch.cat([d[0] for d in reaf_data], dim=0)
        a_all           = torch.cat([d[1] for d in reaf_data], dim=0)
        dz_all          = torch.cat([d[2] for d in reaf_data], dim=0)
        pred_all = agent.latent_stack.reafference_predictor(z_world_raw_all, a_all)
        pred_test = pred_all[n_train:]
        dz_test   = dz_all[n_train:]
        if pred_test.shape[0] == 0:
            return 0.0
        ss_res = ((dz_test - pred_test) ** 2).sum()
        ss_tot = ((dz_test - dz_test.mean(dim=0, keepdim=True)) ** 2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())

    print(f"  reafference R² (test n={pred_test.shape[0]}): {r2:.4f}", flush=True)
    return max(0.0, r2)


# Transition types treated as harm events (including approach)
HARM_TTYPES = {"env_caused_hazard", "agent_caused_hazard", "hazard_approach"}


def _train(
    agent: REEAgent,
    env,
    optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    reaf_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    agent.train()

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF_EACH = 2000

    reaf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_REAF_DATA = 5000

    counts = {t: 0 for t in ["hazard_approach", "env_caused_hazard", "agent_caused_hazard", "none"]}

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_raw_prev = None
        a_prev     = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()
            z_raw_curr   = latent.z_world_raw.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if ttype in counts:
                counts[ttype] += 1

            # harm_eval training: pos includes approach AND contact
            is_harm = harm_signal < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF_EACH:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF_EACH:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF_EACH:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF_EACH:]

            # Reafference data: empty locomotion steps only
            if (ttype == "none" and z_raw_prev is not None and a_prev is not None):
                dz_raw = z_raw_curr - z_raw_prev
                reaf_data.append((z_raw_prev.cpu(), a_prev.cpu(), dz_raw.cpu()))
                if len(reaf_data) > MAX_REAF_DATA:
                    reaf_data = reaf_data[-MAX_REAF_DATA:]

            # Standard E1 + E2_self losses
            e1_loss      = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()
            total_loss   = e1_loss + e2_self_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # ReafferencePredictor loss
            if len(reaf_data) >= 16 and agent.latent_stack.reafference_predictor is not None:
                k = min(32, len(reaf_data))
                idxs = torch.randperm(len(reaf_data))[:k].tolist()
                zwr_b = torch.cat([reaf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([reaf_data[i][1] for i in idxs]).to(agent.device)
                dz_b  = torch.cat([reaf_data[i][2] for i in idxs]).to(agent.device)
                pred_dz = agent.latent_stack.reafference_predictor(zwr_b, a_b)
                reaf_loss = F.mse_loss(pred_dz, dz_b)
                if reaf_loss.requires_grad:
                    reaf_optimizer.zero_grad()
                    reaf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.latent_stack.reafference_predictor.parameters(), 0.5
                    )
                    reaf_optimizer.step()

            # E3 harm_eval balanced training
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
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            z_raw_prev = z_raw_curr
            a_prev     = action.detach()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  approach={counts['hazard_approach']}  "
                f"contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}  "
                f"none={counts['none']}",
                flush=True,
            )

    return {"counts": counts, "reaf_data": reaf_data}


def _eval(agent: REEAgent, env, num_episodes: int, steps_per_episode: int) -> Dict:
    agent.eval()
    scores: Dict[str, List[float]] = {
        "none": [],
        "env_caused_hazard": [],
        "agent_caused_hazard": [],
        "hazard_approach": [],
        "benefit_approach": [],
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
                z_world_corr = latent.z_world

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

    means = {k: float(sum(v) / max(1, len(v))) for k, v in scores.items()}
    pred_std = float(torch.tensor(all_scores).std().item()) if len(all_scores) > 1 else 0.0
    n_counts = {k: len(v) for k, v in scores.items()}

    calibration_gap_approach = means["hazard_approach"] - means["none"]
    calibration_gap_contact  = (
        means["env_caused_hazard"] + means["agent_caused_hazard"]
    ) / 2.0 - means["none"]

    print(
        f"  E3 harm_eval — none={means['none']:.4f}  "
        f"approach={means['hazard_approach']:.4f}  "
        f"contact={means['env_caused_hazard']:.4f}/{means['agent_caused_hazard']:.4f}",
        flush=True,
    )
    print(
        f"  gap_approach={calibration_gap_approach:.4f}  "
        f"gap_contact={calibration_gap_contact:.4f}  "
        f"pred_std={pred_std:.4f}  n={n_counts}",
        flush=True,
    )

    return {
        "means": means,
        "n_counts": n_counts,
        "calibration_gap_approach": calibration_gap_approach,
        "calibration_gap_contact":  calibration_gap_contact,
        "harm_pred_std": pred_std,
        "fatal_errors": fatal_errors,
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
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,   # 12 (includes harm_exposure, benefit_exposure)
        world_obs_dim=env.world_obs_dim, # 250 (includes hazard_field_view, resource_field_view)
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,  # SD-007 enabled
    )
    agent = REEAgent(config)

    assert agent.latent_stack.reafference_predictor is not None
    print(
        f"[V3-EXQ-029] CausalGridWorldV2: body_obs={env.body_obs_dim} world_obs={env.world_obs_dim}",
        flush=True,
    )
    print(
        f"[V3-EXQ-029] SD-007 (MECH-101): ReafferencePredictor(z_world_raw_prev[{world_dim}]+a[{env.action_dim}]→{world_dim})",
        flush=True,
    )
    print(
        f"[V3-EXQ-029] SD-008: alpha_world={alpha_world}  proximity_scale={proximity_scale}",
        flush=True,
    )

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n and "reafference_predictor" not in n
    ]
    reaf_params      = list(agent.latent_stack.reafference_predictor.parameters())
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer       = optim.Adam(standard_params, lr=lr)
    reaf_optimizer  = optim.Adam(reaf_params,     lr=1e-3)
    harm_eval_optim = optim.Adam(harm_eval_params, lr=1e-4)

    train_out = _train(
        agent, env, optimizer, harm_eval_optim, reaf_optimizer,
        warmup_episodes, steps_per_episode,
    )
    reafference_r2 = _compute_reafference_r2(agent, train_out["reaf_data"])

    print(f"[V3-EXQ-029] Eval ({eval_episodes} eps)...", flush=True)
    eval_out = _eval(agent, env, eval_episodes, steps_per_episode)

    c1_pass = eval_out["calibration_gap_approach"] > 0.08
    c2_pass = eval_out["calibration_gap_contact"]  > 0.05
    c3_pass = reafference_r2 > 0.10
    c4_pass = eval_out["harm_pred_std"] > 0.01
    c5_pass = eval_out["n_counts"].get("hazard_approach", 0) >= 30

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: gap_approach={eval_out['calibration_gap_approach']:.4f} <= 0.08"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: gap_contact={eval_out['calibration_gap_contact']:.4f} <= 0.05"
        )
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: reafference_r2={reafference_r2:.4f} <= 0.10")
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: harm_pred_std={eval_out['harm_pred_std']:.4f} <= 0.01 (collapsed)"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_hazard_approach={eval_out['n_counts'].get('hazard_approach', 0)} < 30"
        )

    print(f"\nV3-EXQ-029 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    means = eval_out["means"]
    n_counts = eval_out["n_counts"]
    tc = train_out["counts"]

    metrics = {
        "alpha_world":              float(alpha_world),
        "proximity_scale":          float(proximity_scale),
        "reafference_r2":           float(reafference_r2),
        "calibration_gap_approach": float(eval_out["calibration_gap_approach"]),
        "calibration_gap_contact":  float(eval_out["calibration_gap_contact"]),
        "harm_pred_std":            float(eval_out["harm_pred_std"]),
        "mean_score_none":          float(means["none"]),
        "mean_score_approach":      float(means["hazard_approach"]),
        "mean_score_env_contact":   float(means["env_caused_hazard"]),
        "mean_score_agent_contact": float(means["agent_caused_hazard"]),
        "n_none_eval":              float(n_counts.get("none", 0)),
        "n_approach_eval":          float(n_counts.get("hazard_approach", 0)),
        "n_env_hazard_eval":        float(n_counts.get("env_caused_hazard", 0)),
        "n_agent_hazard_eval":      float(n_counts.get("agent_caused_hazard", 0)),
        "train_approach_events":    float(tc.get("hazard_approach", 0)),
        "train_contact_events":     float(tc.get("env_caused_hazard", 0) + tc.get("agent_caused_hazard", 0)),
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

    summary_markdown = f"""# V3-EXQ-029 — SD-003 on CausalGridWorldV2: Proxy-Gradient World

**Status:** {status}
**Claims:** SD-003, SD-007, ARC-024, MECH-071
**World:** CausalGridWorldV2 (use_proxy_fields=True, proximity_scale={proximity_scale})
**SD-007 (MECH-101):** ReafferencePredictor(z_world_raw_prev[{world_dim}] + a[{env.action_dim}] → {world_dim})
**alpha_world:** {alpha_world}  (SD-008)
**Seed:** {seed}

## Motivation (ARC-024)

Harm/benefit signals are proxies along gradients toward asymptotic limits. A world
generating harm only at contact models the endpoint, not the gradient. CausalGridWorldV2
generates observable gradient fields preceding contact events:
- hazard_field_view (25 channels) in world_obs: visible proximity to hazards
- harm_exposure (EMA) in body_obs: accumulated nociceptive history
- harm_signal at approach steps: continuous gradient signal before contact

HYPOTHESIS: E3.harm_eval trained on gradient-world data fires at approach, not just contact.

## Key Metrics

| Transition Type | mean E3.harm_eval | n |
|---|---|---|
| none (locomotion)        | {means['none']:.4f} | {n_counts.get('none', 0)} |
| hazard_approach (NEW)    | {means['hazard_approach']:.4f} | {n_counts.get('hazard_approach', 0)} |
| env_caused_hazard        | {means['env_caused_hazard']:.4f} | {n_counts.get('env_caused_hazard', 0)} |
| agent_caused_hazard      | {means['agent_caused_hazard']:.4f} | {n_counts.get('agent_caused_hazard', 0)} |

- **gap_approach** (approach - none): **{eval_out['calibration_gap_approach']:.4f}**  (PASS > 0.08)
- **gap_contact** (contact - none):   **{eval_out['calibration_gap_contact']:.4f}**  (PASS > 0.05)
- reafference_r2: **{reafference_r2:.4f}**  (PASS > 0.10)
- harm_pred_std: {eval_out['harm_pred_std']:.4f}

## Training Counts

- approach events: {tc.get('hazard_approach', 0)}
- contact events:  {tc.get('env_caused_hazard', 0) + tc.get('agent_caused_hazard', 0)}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: gap_approach > 0.08 (E3 detects approach) | {"PASS" if c1_pass else "FAIL"} | {eval_out['calibration_gap_approach']:.4f} |
| C2: gap_contact > 0.05 (E3 still detects contact) | {"PASS" if c2_pass else "FAIL"} | {eval_out['calibration_gap_contact']:.4f} |
| C3: reafference_r2 > 0.10 | {"PASS" if c3_pass else "FAIL"} | {reafference_r2:.4f} |
| C4: harm_pred_std > 0.01 | {"PASS" if c4_pass else "FAIL"} | {eval_out['harm_pred_std']:.4f} |
| C5: n_hazard_approach >= 30 | {"PASS" if c5_pass else "FAIL"} | {n_counts.get('hazard_approach', 0)} |

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
        "fatal_error_count": int(eval_out["fatal_errors"]),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--warmup",         type=int,   default=500)
    parser.add_argument("--eval-eps",       type=int,   default=50)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    parser.add_argument("--alpha-self",     type=float, default=0.3)
    parser.add_argument("--harm-scale",     type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_scale=args.proximity_scale,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"]  = CLAIM_IDS[0]
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
