"""
V3-EXQ-039 — Training Progression Analysis (EXQ-033 Follow-up)

Claims: MECH-071, ARC-024

EXQ-033 FAIL analysis:

    C4 criterion failed: approach_slope <= contact_slope (0.000265 vs 0.000289).
    Expected: E3 should learn to detect hazard approach (gradient signal) earlier
    and faster than contact (harder signal, requires E2 counterfactual reasoning).

    Two interpretations:
    (a) Contact transitions are more frequent and produce stronger harm_signal,
        so E3 learns them faster despite being "later" in the causal chain.
    (b) The EXQ-033 slope measurement is too noisy at the checkpoint resolution
        (4 checkpoints: 100, 250, 500, 1000) to reliably detect slope differences.

This experiment (EXQ-039) addresses (b) by using finer-grained checkpointing:
    - Log calibration_gap every 50 episodes (vs 150/250/500 in EXQ-033)
    - 800 total training episodes (vs 1000, slightly shorter for speed)
    - 16 checkpoints total → much more reliable slope estimation

If approach_slope > contact_slope holds: validates the "love expands" derivation —
gradient-based harm detection precedes contact-based detection in learning.
If approach_slope <= contact_slope: confirms EXQ-033 result and indicates the
training objective itself needs revision (contact provides stronger gradient signal).

Design:
    - CausalGridWorldV2 (same world as EXQ-026, EXQ-029)
    - Train 800 episodes total
    - Every 50 episodes: eval 20 eps, record calibration_gap_approach and calibration_gap_contact
    - Compute slope via linear regression over all 16 checkpoints
    - Also track when approach first exceeds 0.01 and contact first exceeds 0.01

PASS criteria (ALL must hold):
    C1: final calibration_gap_approach > 0.03
        (E3 learns approach detection with sufficient magnitude)
    C2: calibration_gap_approach at checkpoint 2 (ep 100) > calibration_gap_approach at ep 50
        (monotone improvement in at least 1 early pair — learning has started)
    C3: approach_slope > 0 (positive trend across full run)
    C4: approach_slope > contact_slope  (approach calibration improves FASTER than contact)
    C5: n_approach_min >= 20  (minimum approach events at any checkpoint)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_039_training_progression"
CLAIM_IDS = ["MECH-071", "ARC-024"]
CHECKPOINT_INTERVAL = 50  # episodes between calibration checkpoints
EVAL_EPISODES_PER_CHECKPOINT = 20
TOTAL_TRAIN_EPISODES = 800


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _eval_calibration_gap(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
) -> Tuple[float, float, int, int]:
    """
    Quick calibration eval. Returns:
        (calibration_gap_approach, calibration_gap_contact, n_approach, n_contact)
    """
    agent.eval()

    scores: Dict[str, List[float]] = {
        "none": [], "hazard_approach": [],
        "env_caused_hazard": [], "agent_caused_hazard": [],
    }

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            with torch.no_grad():
                score = float(agent.e3.harm_eval(z_world).item())
            if ttype in scores:
                scores[ttype].append(score)

            if done:
                break

    agent.train()

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    mean_none     = _mean(scores["none"])
    mean_approach = _mean(scores["hazard_approach"])
    mean_contact  = _mean(
        scores["env_caused_hazard"] + scores["agent_caused_hazard"]
    )

    cal_gap_approach = mean_approach - mean_none
    cal_gap_contact  = mean_contact  - mean_none
    n_approach = len(scores["hazard_approach"])
    n_contact  = len(scores["env_caused_hazard"]) + len(scores["agent_caused_hazard"])

    return cal_gap_approach, cal_gap_contact, n_approach, n_contact


def run(
    seed: int = 0,
    total_train_episodes: int = TOTAL_TRAIN_EPISODES,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    eval_episodes: int = EVAL_EPISODES_PER_CHECKPOINT,
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
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)

    print(
        f"[V3-EXQ-039] Training Progression Analysis\n"
        f"  total_episodes={total_train_episodes}  checkpoint_interval={checkpoint_interval}\n"
        f"  eval_per_checkpoint={eval_episodes}  steps={steps_per_episode}\n"
        f"  alpha_world={alpha_world}",
        flush=True,
    )

    # Separate optimizers (MECH-069)
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    wf_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    he_params = list(agent.e3.harm_eval_head.parameters())

    optimizer    = optim.Adam(standard_params, lr=lr)
    wf_optimizer = optim.Adam(wf_params,       lr=1e-3)
    he_optimizer = optim.Adam(he_params,       lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000
    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000
    num_actions = env.action_dim

    # Checkpoints: (episode, gap_approach, gap_contact, n_approach, n_contact)
    checkpoints: List[Dict] = []

    agent.train()

    for ep in range(total_train_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        a_prev = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if harm_signal < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            loss = e1_loss + e2_loss
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(wf_data) >= 16:
                k = min(32, len(wf_data))
                idxs = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e2.world_transition.parameters(), 0.5)
                    wf_optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0).to(agent.device)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0).to(agent.device)
                zw_b   = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    he_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    he_optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()

            if done:
                break

        # Checkpoint eval every `checkpoint_interval` episodes
        if (ep + 1) % checkpoint_interval == 0:
            gap_a, gap_c, n_a, n_c = _eval_calibration_gap(
                agent, env, eval_episodes, steps_per_episode
            )
            checkpoints.append({
                "episode":       ep + 1,
                "gap_approach":  gap_a,
                "gap_contact":   gap_c,
                "n_approach":    n_a,
                "n_contact":     n_c,
            })
            print(
                f"  [ep {ep+1:4d}] gap_approach={gap_a:.4f}  gap_contact={gap_c:.4f}  "
                f"n_approach={n_a}  n_contact={n_c}",
                flush=True,
            )

    # --- Slope computation via linear regression ---
    episodes_arr = np.array([c["episode"] for c in checkpoints], dtype=float)
    gap_a_arr    = np.array([c["gap_approach"] for c in checkpoints], dtype=float)
    gap_c_arr    = np.array([c["gap_contact"]  for c in checkpoints], dtype=float)

    def _slope(x, y):
        """Linear regression slope (y ~ x)."""
        if len(x) < 2:
            return 0.0
        x_c = x - x.mean()
        y_c = y - y.mean()
        denom = float((x_c * x_c).sum())
        if denom < 1e-12:
            return 0.0
        return float((x_c * y_c).sum() / denom)

    approach_slope = _slope(episodes_arr, gap_a_arr)
    contact_slope  = _slope(episodes_arr, gap_c_arr)

    final_gap_approach = float(gap_a_arr[-1]) if len(gap_a_arr) > 0 else 0.0
    final_n_approach   = checkpoints[-1]["n_approach"] if checkpoints else 0

    # C2: early monotone improvement (checkpoint 1 → 2)
    c2_val = (
        (checkpoints[1]["gap_approach"] > checkpoints[0]["gap_approach"])
        if len(checkpoints) >= 2 else False
    )

    n_approach_min = min((c["n_approach"] for c in checkpoints), default=0)

    c1_pass = final_gap_approach > 0.03
    c2_pass = bool(c2_val)
    c3_pass = approach_slope > 0
    c4_pass = approach_slope > contact_slope
    c5_pass = n_approach_min >= 20

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(f"C1 FAIL: final_gap_approach={final_gap_approach:.4f} <= 0.03")
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: gap_approach did not improve ep{checkpoints[0]['episode']}→{checkpoints[1]['episode']}"
            if len(checkpoints) >= 2 else "C2 FAIL: fewer than 2 checkpoints"
        )
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: approach_slope={approach_slope:.6f} <= 0")
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: approach_slope({approach_slope:.6f}) <= contact_slope({contact_slope:.6f})"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: n_approach_min={n_approach_min} < 20")

    print(f"\n[V3-EXQ-039] Slopes: approach={approach_slope:.6f}  contact={contact_slope:.6f}", flush=True)
    print(f"V3-EXQ-039 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "approach_slope":        float(approach_slope),
        "contact_slope":         float(contact_slope),
        "slope_diff_a_minus_c":  float(approach_slope - contact_slope),
        "final_gap_approach":    float(final_gap_approach),
        "final_gap_contact":     float(gap_c_arr[-1]) if len(gap_c_arr) > 0 else 0.0,
        "n_approach_min":        float(n_approach_min),
        "num_checkpoints":       float(len(checkpoints)),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(n_met),
    }
    # Per-checkpoint metrics
    for cp in checkpoints:
        prefix = f"ep{cp['episode']}"
        metrics[f"{prefix}_gap_approach"] = float(cp["gap_approach"])
        metrics[f"{prefix}_gap_contact"]  = float(cp["gap_contact"])
        metrics[f"{prefix}_n_approach"]   = float(cp["n_approach"])

    # Checkpoint table for markdown
    cp_rows = ""
    for cp in checkpoints:
        cp_rows += (
            f"| {cp['episode']:4d} | {cp['gap_approach']:.4f} | {cp['gap_contact']:.4f} "
            f"| {cp['n_approach']:4d} | {cp['n_contact']:4d} |\n"
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-039 — Training Progression Analysis

**Status:** {status}
**Claims:** MECH-071, ARC-024
**World:** CausalGridWorldV2 (proximity_scale={proximity_scale})
**alpha_world:** {alpha_world}  (SD-008)
**Seed:** {seed}
**Predecessor:** EXQ-033 (FAIL — C4: approach_slope ≤ contact_slope with 4 checkpoints)

## Motivation

EXQ-033 used 4 checkpoints (100, 250, 500, 1000 episodes) and found approach_slope=0.000265
vs contact_slope=0.000289. The hierarchical prediction is that approach detection should
emerge faster (gradient signal is observable before contact). EXQ-039 tests with 16
checkpoints (every 50 episodes over 800 total) for much more reliable slope estimation.

## Checkpoint Progression

| Episode | gap_approach | gap_contact | n_approach | n_contact |
|---|---|---|---|---|
{cp_rows}
**approach_slope:** {approach_slope:.6f}  **contact_slope:** {contact_slope:.6f}
**slope_diff (approach − contact):** {approach_slope - contact_slope:.6f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: final_gap_approach > 0.03 | {"PASS" if c1_pass else "FAIL"} | {final_gap_approach:.4f} |
| C2: early monotone improvement (ep50 → ep100) | {"PASS" if c2_pass else "FAIL"} | {"yes" if c2_pass else "no"} |
| C3: approach_slope > 0 | {"PASS" if c3_pass else "FAIL"} | {approach_slope:.6f} |
| C4: approach_slope > contact_slope | {"PASS" if c4_pass else "FAIL"} | {approach_slope:.6f} vs {contact_slope:.6f} |
| C5: n_approach_min >= 20 | {"PASS" if c5_pass else "FAIL"} | {n_approach_min} |

Criteria met: {n_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if n_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",               type=int,   default=0)
    parser.add_argument("--total-episodes",     type=int,   default=TOTAL_TRAIN_EPISODES)
    parser.add_argument("--checkpoint-interval", type=int,  default=CHECKPOINT_INTERVAL)
    parser.add_argument("--eval-eps",           type=int,   default=EVAL_EPISODES_PER_CHECKPOINT)
    parser.add_argument("--steps",              type=int,   default=200)
    parser.add_argument("--alpha-world",        type=float, default=0.9)
    parser.add_argument("--harm-scale",         type=float, default=0.02)
    parser.add_argument("--proximity-scale",    type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        total_train_episodes=args.total_episodes,
        checkpoint_interval=args.checkpoint_interval,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
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
