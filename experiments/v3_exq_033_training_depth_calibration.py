"""
V3-EXQ-033 — Training Depth vs. Gradient Detection Depth
         "Love Expands Under Intelligence" Computational Test

Claims: ARC-024, MECH-071, INV-029

Philosophical basis (Philosophy/notes/2026-03-18_love_once_means_love_all.md):
    "With increasing intelligence, the agent's causal model deepens. But the scope
    of what it cannot be sure about expands faster than its certainty grows."
    The derivation predicts that love (care/harm-avoidance) enters as a local
    commitment and EXPANDS under intelligence and uncertainty.

    Computational instantiation: harm detection extends further back along the
    causal chain with more training. Early training: only contact events reliably
    detected. Later training: approach detected. Even later: the gradient extends
    further back, covering more of the causal structure.

    EXQ-029 showed the endpoint (500 eps → calibration_gap_approach=0.239).
    This experiment tests THE CURVE — whether gradient detection deepens
    monotonically and whether approach (one step back from contact) improves
    faster than contact (the endpoint) across training.

    If the curve shows approach_slope > contact_slope, it is the first direct
    computational evidence that intelligent harm-avoidance extends backward
    along causal chains — the mechanism of "love expands."

Design:
    Train for 1000 episodes total. Pause training at checkpoints [100, 250, 500, 1000]
    and run a lightweight eval (30 eps) at each. Record calibration_gap_approach
    and calibration_gap_contact at each checkpoint.

    calibration_gap_approach = E3(hazard_approach) - E3(none)  [detects gradient step]
    calibration_gap_contact  = E3(contact) - E3(none)          [detects endpoint]

    The derivation predicts: approach_gap grows MORE than contact_gap across training,
    because the gradient (approach detection) requires learning a richer causal model
    than the endpoint (contact detection), and intelligence compounds this advantage.

PASS criteria (ALL must hold):
    C1: calibration_gap_approach[1000] > calibration_gap_approach[100] + 0.05
        Approach detection grows with training (not noise-floor throughout)
    C2: calibration_gap_approach[500] > calibration_gap_approach[100]
        Monotonic in first half of training
    C3: calibration_gap_approach[1000] > 0.10
        Non-trivial gradient detection at endpoint
    C4: approach_slope > contact_slope
        Core claim: approach detection improves FASTER than contact detection.
        slope = (gap[1000] - gap[100]) / 900  (per-episode rate)
        This is the computational signature of the philosophical derivation.
    C5: n_approach_steps >= 20 in each checkpoint eval
        Sufficient approach events at each stage
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


EXPERIMENT_TYPE = "v3_exq_033_training_depth_calibration"
CLAIM_IDS = ["ARC-024", "MECH-071", "INV-029"]

CHECKPOINTS = [100, 250, 500, 1000]   # total episodes at each eval pause
EVAL_EPISODES = 30                      # lightweight eval at each checkpoint


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _train_n_episodes(
    agent: REEAgent,
    env,
    optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    reaf_optimizer: optim.Optimizer,
    n_episodes: int,
    steps_per_episode: int,
    harm_buf_pos: List,
    harm_buf_neg: List,
    reaf_data: List,
    counts: Dict,
    episode_offset: int = 0,
) -> None:
    """
    Train for n_episodes, accumulating into shared buffers.
    Modifies harm_buf_pos, harm_buf_neg, reaf_data, counts in-place.
    """
    MAX_BUF = 2000
    MAX_REAF = 5000
    agent.train()

    for ep in range(n_episodes):
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
            counts[ttype] = counts.get(ttype, 0) + 1

            if harm_signal < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos[:] = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg[:] = harm_buf_neg[-MAX_BUF:]

            if ttype == "none" and z_raw_prev is not None and a_prev is not None:
                dz_raw = z_raw_curr - z_raw_prev
                reaf_data.append((z_raw_prev.cpu(), a_prev.cpu(), dz_raw.cpu()))
                if len(reaf_data) > MAX_REAF:
                    reaf_data[:] = reaf_data[-MAX_REAF:]

            # Standard losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # Reafference
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

            # harm_eval training (balanced, observed states only — E2-predicted
            # states would require separate wf training; keeping this pure)
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

        global_ep = episode_offset + ep + 1
        if global_ep % 100 == 0 or global_ep == episode_offset + n_episodes:
            approach = counts.get("hazard_approach", 0)
            contact  = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"  [train] ep {global_ep}  approach={approach}  contact={contact}  "
                f"pos_buf={len(harm_buf_pos)}  neg_buf={len(harm_buf_neg)}",
                flush=True,
            )


def _eval_checkpoint(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
    checkpoint_eps: int,
) -> Dict:
    """Lightweight eval — measure calibration_gap_approach and calibration_gap_contact."""
    agent.eval()
    scores: Dict[str, List[float]] = {
        "none": [], "env_caused_hazard": [], "agent_caused_hazard": [],
        "hazard_approach": [], "benefit_approach": [],
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
                score = float(agent.e3.harm_eval(z_world).item())

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action
            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in scores:
                scores[ttype].append(score)
            if done:
                break

    means = {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}
    n_counts = {k: len(v) for k, v in scores.items()}

    gap_approach = means["hazard_approach"] - means["none"]
    contact_mean = (means["env_caused_hazard"] * n_counts["env_caused_hazard"] +
                    means["agent_caused_hazard"] * n_counts["agent_caused_hazard"])
    contact_n = n_counts["env_caused_hazard"] + n_counts["agent_caused_hazard"]
    gap_contact  = (contact_mean / contact_n - means["none"]) if contact_n > 0 else 0.0

    print(
        f"  [eval @{checkpoint_eps}ep] "
        f"none={means['none']:.3f}  approach={means['hazard_approach']:.3f}  "
        f"contact={means['env_caused_hazard']:.3f}  "
        f"gap_approach={gap_approach:.4f}  gap_contact={gap_contact:.4f}  "
        f"n_approach={n_counts['hazard_approach']}",
        flush=True,
    )

    return {
        "episodes": checkpoint_eps,
        "mean_none":           means["none"],
        "mean_approach":       means["hazard_approach"],
        "mean_env_hazard":     means["env_caused_hazard"],
        "mean_agent_hazard":   means["agent_caused_hazard"],
        "calibration_gap_approach": gap_approach,
        "calibration_gap_contact":  gap_contact,
        "n_approach":          n_counts["hazard_approach"],
        "n_contact":           contact_n,
    }


def run(
    seed: int = 0,
    max_episodes: int = 1000,
    eval_episodes: int = 30,
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
        f"[V3-EXQ-033] Training Depth vs. Gradient Detection Depth\n"
        f"  'Love Expands Under Intelligence' Computational Test\n"
        f"  checkpoints={CHECKPOINTS}  eval_eps={eval_episodes}\n"
        f"  body_obs={env.body_obs_dim}  world_obs={env.world_obs_dim}  alpha_world={alpha_world}",
        flush=True,
    )

    # Three separate optimizers (MECH-069)
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    reaf_params = (
        list(agent.latent_stack.reafference_predictor.parameters())
        if agent.latent_stack.reafference_predictor is not None else []
    )

    optimizer           = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)
    reaf_optimizer      = optim.Adam(reaf_params, lr=1e-3) if reaf_params else optim.Adam(
        [torch.zeros(1, requires_grad=True)], lr=1e-3
    )

    # Shared training state
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    reaf_data:   List[Tuple]         = []
    counts:      Dict[str, int]      = {}

    # Train with checkpoint evals
    curve: List[Dict] = []
    prev_checkpoint = 0

    for checkpoint in CHECKPOINTS:
        n_to_train = checkpoint - prev_checkpoint
        print(f"\n[V3-EXQ-033] Training episodes {prev_checkpoint+1}–{checkpoint}...", flush=True)

        _train_n_episodes(
            agent, env, optimizer, harm_eval_optimizer, reaf_optimizer,
            n_to_train, steps_per_episode,
            harm_buf_pos, harm_buf_neg, reaf_data, counts,
            episode_offset=prev_checkpoint,
        )

        print(f"\n[V3-EXQ-033] Checkpoint eval @ {checkpoint} eps ({eval_episodes} ep)...", flush=True)
        result = _eval_checkpoint(agent, env, eval_episodes, steps_per_episode, checkpoint)
        curve.append(result)
        prev_checkpoint = checkpoint

    # Compute slopes and criteria
    gap_approach_100  = curve[0]["calibration_gap_approach"]
    gap_approach_500  = curve[2]["calibration_gap_approach"]
    gap_approach_1000 = curve[3]["calibration_gap_approach"]
    gap_contact_100   = curve[0]["calibration_gap_contact"]
    gap_contact_1000  = curve[3]["calibration_gap_contact"]
    n_approach_min    = min(c["n_approach"] for c in curve)

    approach_slope = (gap_approach_1000 - gap_approach_100) / 900.0
    contact_slope  = (gap_contact_1000  - gap_contact_100)  / 900.0

    c1_pass = gap_approach_1000 > gap_approach_100 + 0.05
    c2_pass = gap_approach_500  > gap_approach_100
    c3_pass = gap_approach_1000 > 0.10
    c4_pass = approach_slope    > contact_slope
    c5_pass = n_approach_min    >= 20

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: gap_approach[1000]({gap_approach_1000:.4f}) <= gap_approach[100]({gap_approach_100:.4f}) + 0.05"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: gap_approach[500]({gap_approach_500:.4f}) <= gap_approach[100]({gap_approach_100:.4f})"
        )
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: gap_approach[1000]={gap_approach_1000:.4f} <= 0.10")
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: approach_slope({approach_slope:.6f}/ep) <= contact_slope({contact_slope:.6f}/ep)"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: n_approach_min={n_approach_min} < 20")

    print(f"\nV3-EXQ-033 verdict: {status}  ({n_met}/5)", flush=True)
    print(f"  approach_slope={approach_slope:.6f}/ep  contact_slope={contact_slope:.6f}/ep", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "alpha_world":              float(alpha_world),
        "proximity_scale":          float(proximity_scale),
        "approach_slope_per_ep":    float(approach_slope),
        "contact_slope_per_ep":     float(contact_slope),
        "slope_ratio":              float(approach_slope / (contact_slope + 1e-8)),
        "n_approach_min":           float(n_approach_min),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(n_met),
        "fatal_error_count": 0.0,
    }
    for c in curve:
        ep = c["episodes"]
        metrics[f"gap_approach_ep{ep}"] = float(c["calibration_gap_approach"])
        metrics[f"gap_contact_ep{ep}"]  = float(c["calibration_gap_contact"])
        metrics[f"n_approach_ep{ep}"]   = float(c["n_approach"])

    curve_rows = "\n".join(
        f"| {c['episodes']:4d} | {c['mean_none']:.3f} | {c['mean_approach']:.3f} | {c['mean_env_hazard']:.3f} | "
        f"{c['calibration_gap_approach']:.4f} | {c['calibration_gap_contact']:.4f} | {c['n_approach']} |"
        for c in curve
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-033 — Training Depth vs. Gradient Detection Depth
            "Love Expands Under Intelligence" Computational Test

**Status:** {status}
**Claims:** ARC-024, MECH-071, INV-029
**World:** CausalGridWorldV2 (proximity_scale={proximity_scale})
**alpha_world:** {alpha_world}  (SD-008)
**Seed:** {seed}

## Philosophical Basis

From Philosophy/notes/2026-03-18_love_once_means_love_all.md:
    "Love enters as a point commitment and expands, under intelligence and uncertainty,
    until it covers everything."

Computational prediction: with increasing training (capability), E3 detects harm
gradients further back along the causal chain. Approach detection (one step from contact)
should improve faster than contact detection (the endpoint), because the gradient
representation requires a richer causal model.

approach_slope > contact_slope → gradient extends backward with intelligence.

## Calibration Curve

| eps | none | approach | contact | gap_approach | gap_contact | n_approach |
|---|---|---|---|---|---|---|
{curve_rows}

- **Approach slope**: {approach_slope:.6f} per episode
- **Contact slope**:  {contact_slope:.6f} per episode
- **Slope ratio** (approach/contact): {approach_slope / (contact_slope + 1e-8):.3f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: gap_approach[1000] > gap_approach[100] + 0.05 (grows with training) | {"PASS" if c1_pass else "FAIL"} | {gap_approach_1000:.4f} vs {gap_approach_100:.4f} |
| C2: gap_approach[500] > gap_approach[100] (monotonic in first half) | {"PASS" if c2_pass else "FAIL"} | {gap_approach_500:.4f} vs {gap_approach_100:.4f} |
| C3: gap_approach[1000] > 0.10 (non-trivial at endpoint) | {"PASS" if c3_pass else "FAIL"} | {gap_approach_1000:.4f} |
| C4: approach_slope > contact_slope (CORE — gradient deepens faster) | {"PASS" if c4_pass else "FAIL"} | {approach_slope:.6f} vs {contact_slope:.6f} |
| C5: n_approach_min >= 20 (sufficient events at each checkpoint) | {"PASS" if c5_pass else "FAIL"} | {n_approach_min} |

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
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--max-episodes",    type=int,   default=1000)
    parser.add_argument("--eval-eps",        type=int,   default=30)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        max_episodes=args.max_episodes,
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
