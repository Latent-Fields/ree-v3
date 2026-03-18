"""
V3-EXQ-015 — Three-Stream Lateral Head (MECH-099)

Claims: MECH-099 (three-pathway architecture), SD-003 (causal attribution).

Motivation (2026-03-17):
  The biological three-stream architecture (Haak & Beckmann 2018, HCP n=470)
  shows a lateral/third stream (MT→MST/FST→STS) specialised for dynamic motion
  and biological motion — it terminates near TPJ and feeds harm/agency detection
  directly, bypassing ventral (content) processing.

  In REE terms: E3's harm_eval operates on z_world, which is contaminated by
  perspective shift (E2 identity shortcut). A dedicated lateral encoder head
  processing ONLY hazard + contamination channels gives E3 a harm-salient
  embedding that is:
    1. Invariant to perspective shift (same hazard/contamination state regardless
       of which direction the agent moved)
    2. Specific to the harm-relevant features (entity type=hazard, contamination)
    3. Lower-dimensional (harm_dim=16 vs world_dim=32)

  This experiment tests whether z_harm (from lateral head) produces a valid
  SD-003 calibration_gap > 0.05 WITHOUT reafference correction.

  SD-003 attribution pipeline (lateral variant):
    z_harm_actual = encode(world_obs with actual action → lateral head → z_harm)
    z_harm_cf     = encode(world_obs + cf_action → direct probe on saved z_harm)
    causal_sig    = net_eval(z_harm_actual) - net_eval(z_harm_cf)

  Implementation note: Because z_harm is derived from world_obs (not from
  z_world + action through E2), the counterfactual comparison requires probing
  from near-hazard positions where action choice matters: moving INTO the hazard
  vs moving away. The lateral head reads the current local_view's hazard/contamination
  channels, so the difference between positions (near vs far from hazard) is the
  causal signal.

PASS criteria (ALL must hold):
  C1: calibration_gap > 0.05 in lateral attribution test (near-hazard z_harm vs safe)
  C2: z_harm responds more strongly to hazard-adjacent positions than z_world does
      (mean ||z_harm|| at near-hazard > mean ||z_harm|| at safe, margin > 0.01)
  C3: Warmup benefit events > 20
  C4: n_near_hazard >= 10 AND n_safe >= 10
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


EXPERIMENT_TYPE = "v3_exq_015_three_stream_lateral"
CLAIM_IDS = ["MECH-099", "SD-003"]

HARM_DIM = 16        # lateral head output dimension
E2_ROLLOUT_STEPS = 5
RECON_WEIGHT = 1.0


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _random_cf_action(actual_idx: int, num_actions: int) -> int:
    choices = [a for a in range(num_actions) if a != actual_idx]
    return random.choice(choices) if choices else 0


def _make_world_decoder(world_dim: int, world_obs_dim: int, hidden_dim: int = 64):
    return nn.Sequential(
        nn.Linear(world_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, world_obs_dim),
    )


def _make_net_eval_head(input_dim: int, hidden_dim: int = 32) -> nn.Module:
    """Signed regression net_eval head: z_harm → scalar net value ∈ [-1, 1]."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Tanh(),
    )


def _compute_e2w_loss(agent, traj_buffer, rollout_steps, batch_size=8):
    if len(traj_buffer) < 2:
        return next(agent.e1.parameters()).sum() * 0.0
    n = min(batch_size, len(traj_buffer))
    idxs = torch.randperm(len(traj_buffer))[:n].tolist()
    total = next(agent.e1.parameters()).sum() * 0.0
    count = 0
    for idx in idxs:
        seg = traj_buffer[idx]
        if len(seg) < rollout_steps + 1:
            continue
        z = seg[0][0]
        z_target = seg[rollout_steps][0]
        for k in range(rollout_steps):
            z = agent.e2.world_forward(z, seg[k][1])
        total = total + F.mse_loss(z, z_target)
        count += 1
    return total / count if count > 0 else total


def _train_episodes(
    agent: REEAgent,
    env: CausalGridWorld,
    world_decoder: nn.Module,
    net_eval_head: nn.Module,
    optimizer: optim.Optimizer,
    net_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    harm_scale: float = 0.02,
) -> Dict:
    agent.train()
    world_decoder.train()
    net_eval_head.train()

    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200
    signal_buffer: List[Tuple[torch.Tensor, float]] = []
    MAX_SIGNAL_BUFFER = 1000

    total_harm = 0
    total_benefit = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        episode_traj: List = []

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            episode_traj.append((latent.z_world.detach(), action.detach()))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if harm_signal < 0:
                total_harm += 1
            elif harm_signal > 0:
                total_benefit += 1

            # Collect (z_harm, signal) for net_eval training
            if latent.z_harm is not None:
                if harm_signal != 0.0 or (step % 3 == 0):
                    signal_buffer.append((latent.z_harm.detach(), float(harm_signal)))
                    if len(signal_buffer) > MAX_SIGNAL_BUFFER:
                        signal_buffer = signal_buffer[-MAX_SIGNAL_BUFFER:]

            # E1 + E2_self + E2_world + reconstruction
            e1_loss = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()
            e2w_loss = _compute_e2w_loss(agent, traj_buffer, E2_ROLLOUT_STEPS)

            obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
            z_w = agent.latent_stack.split_encoder.world_encoder(obs_w)
            recon = world_decoder(z_w)
            recon_loss = F.mse_loss(recon, obs_w)

            total_loss = e1_loss + e2_self_loss + e2w_loss + RECON_WEIGHT * recon_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(world_decoder.parameters(), 1.0)
                optimizer.step()

            # net_eval regression on z_harm
            if len(signal_buffer) >= 8:
                k = min(32, len(signal_buffer))
                idxs = torch.randperm(len(signal_buffer))[:k].tolist()
                zw_batch = torch.cat([signal_buffer[i][0] for i in idxs], dim=0)
                sv_batch = torch.tensor(
                    [signal_buffer[i][1] for i in idxs], device=agent.device
                ).unsqueeze(1)
                pred = net_eval_head(zw_batch)
                sv_norm = (sv_batch / harm_scale).clamp(-1.0, 1.0)
                net_loss = F.mse_loss(pred, sv_norm)
                if net_loss.requires_grad:
                    net_eval_optimizer.zero_grad()
                    net_loss.backward()
                    torch.nn.utils.clip_grad_norm_(net_eval_head.parameters(), 0.5)
                    net_eval_optimizer.step()

            if done:
                break

        for start in range(0, len(episode_traj) - E2_ROLLOUT_STEPS):
            traj_buffer.append(episode_traj[start:start + E2_ROLLOUT_STEPS + 1])
        if len(traj_buffer) > MAX_TRAJ_BUFFER:
            traj_buffer = traj_buffer[-MAX_TRAJ_BUFFER:]

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}  "
                  f"benefit={total_benefit}  sig_buf={len(signal_buffer)}", flush=True)

    return {"total_harm": total_harm, "total_benefit": total_benefit}


def _eval_probes(
    agent: REEAgent,
    net_eval_head: nn.Module,
    env: CausalGridWorld,
    num_resets: int,
) -> Dict:
    """
    Probe near-hazard vs safe positions.

    For lateral variant: z_harm comes directly from the encoder at each position.
    Calibration gap = mean(net_eval(z_harm) at near-hazard) - mean(net_eval(z_harm) at safe)
    """
    agent.eval()
    net_eval_head.eval()

    near_vals: List[float] = []
    safe_vals:  List[float] = []
    near_harm_norms: List[float] = []
    safe_harm_norms: List[float] = []
    world_near_norms: List[float] = []
    world_safe_norms: List[float] = []
    all_pred_vals: List[float] = []
    fatal_errors = 0

    wall_type   = env.ENTITY_TYPES["wall"]
    hazard_type = env.ENTITY_TYPES["hazard"]

    def _probe_pos(ax: int, ay: int) -> Tuple[float, float, float]:
        """Returns (net_eval_val, z_harm_norm, z_world_norm) at position ax, ay."""
        env.agent_x = ax
        env.agent_y = ay
        obs_dict = env._get_observation_dict()
        with torch.no_grad():
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            z_harm = latent.z_harm   # [1, harm_dim]
            z_world = latent.z_world # [1, world_dim]
            if z_harm is None:
                return 0.0, 0.0, 0.0
            val = float(net_eval_head(z_harm).item())
            harm_norm = float(torch.norm(z_harm).item())
            world_norm = float(torch.norm(z_world).item())
            return val, harm_norm, world_norm

    try:
        for _ in range(num_resets):
            env.reset()

            # Near-hazard: positions adjacent to hazards
            for hx, hy in env.hazards:
                for action_idx, (dx, dy) in env.ACTIONS.items():
                    if action_idx == 4:
                        continue
                    ax, ay = hx - dx, hy - dy
                    if 0 <= ax < env.size and 0 <= ay < env.size:
                        cell = int(env.grid[ax, ay])
                        if cell not in (wall_type, hazard_type):
                            val, harm_n, world_n = _probe_pos(ax, ay)
                            near_vals.append(val)
                            near_harm_norms.append(harm_n)
                            world_near_norms.append(world_n)
                            all_pred_vals.append(val)

            # Safe: positions far from all hazards (min dist > 3)
            for px in range(env.size):
                for py in range(env.size):
                    if int(env.grid[px, py]) in (wall_type, hazard_type):
                        continue
                    min_dist = min(abs(px - hx) + abs(py - hy)
                                   for hx, hy in env.hazards)
                    if min_dist > 3:
                        val, harm_n, world_n = _probe_pos(px, py)
                        safe_vals.append(val)
                        safe_harm_norms.append(harm_n)
                        world_safe_norms.append(world_n)
                        all_pred_vals.append(val)

    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL probe: {traceback.format_exc()}", flush=True)

    mean_near = float(sum(near_vals) / max(1, len(near_vals)))
    mean_safe = float(sum(safe_vals)  / max(1, len(safe_vals)))
    calibration_gap = mean_near - mean_safe

    mean_harm_near  = float(sum(near_harm_norms)   / max(1, len(near_harm_norms)))
    mean_harm_safe  = float(sum(safe_harm_norms)   / max(1, len(safe_harm_norms)))
    mean_world_near = float(sum(world_near_norms)  / max(1, len(world_near_norms)))
    mean_world_safe = float(sum(world_safe_norms)  / max(1, len(world_safe_norms)))

    pred_std = float(torch.tensor(all_pred_vals).std().item()) if len(all_pred_vals) > 1 else 0.0

    print(f"  Probe — n_near={len(near_vals)} n_safe={len(safe_vals)}", flush=True)
    print(f"  net_eval near={mean_near:.4f}  safe={mean_safe:.4f}  gap={calibration_gap:.4f}",
          flush=True)
    print(f"  ||z_harm|| near={mean_harm_near:.4f}  safe={mean_harm_safe:.4f}  "
          f"margin={mean_harm_near-mean_harm_safe:.4f}", flush=True)
    print(f"  ||z_world|| near={mean_world_near:.4f}  safe={mean_world_safe:.4f}  "
          f"pred_std={pred_std:.4f}", flush=True)

    return {
        "calibration_gap":        calibration_gap,
        "mean_net_eval_near":     mean_near,
        "mean_net_eval_safe":     mean_safe,
        "mean_z_harm_norm_near":  mean_harm_near,
        "mean_z_harm_norm_safe":  mean_harm_safe,
        "z_harm_selectivity_margin": mean_harm_near - mean_harm_safe,
        "mean_z_world_norm_near": mean_world_near,
        "mean_z_world_norm_safe": mean_world_safe,
        "n_near_hazard_probes":   len(near_vals),
        "n_safe_probes":          len(safe_vals),
        "net_eval_pred_std":      pred_std,
        "fatal_errors":           fatal_errors,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 1000,
    eval_probe_resets: int = 10,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    harm_scale: float = 0.02,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(seed=seed, size=12, num_hazards=15, num_resources=5,
                          hazard_harm=harm_scale, contaminated_harm=harm_scale)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        harm_dim=HARM_DIM,  # Enable lateral head (MECH-099)
        alpha_world=alpha_world,
    )
    agent = REEAgent(config)
    world_decoder  = _make_world_decoder(world_dim, env.world_obs_dim)
    net_eval_head  = _make_net_eval_head(HARM_DIM)

    e12_params = list(agent.parameters()) + list(world_decoder.parameters())
    optimizer       = optim.Adam(e12_params, lr=lr)
    net_eval_optim  = optim.Adam(net_eval_head.parameters(), lr=1e-4)

    print(f"[V3-EXQ-015] Warmup: {warmup_episodes} eps, lateral head harm_dim={HARM_DIM}",
          flush=True)
    train_metrics = _train_episodes(
        agent, env, world_decoder, net_eval_head,
        optimizer, net_eval_optim,
        warmup_episodes, steps_per_episode,
        harm_scale=harm_scale,
    )
    warmup_harm    = train_metrics["total_harm"]
    warmup_benefit = train_metrics["total_benefit"]
    print(f"  Warmup done. harm={warmup_harm}  benefit={warmup_benefit}", flush=True)

    print(f"[V3-EXQ-015] Probing ({eval_probe_resets} resets)...", flush=True)
    probe = _eval_probes(agent, net_eval_head, env, eval_probe_resets)
    fatal_errors = probe["fatal_errors"]

    # ── PASS / FAIL ───────────────────────────────────────────────────────
    c1_pass = probe["calibration_gap"] > 0.05
    c2_pass = probe["z_harm_selectivity_margin"] > 0.01
    c3_pass = warmup_benefit > 20
    c4_pass = probe["n_near_hazard_probes"] >= 10 and probe["n_safe_probes"] >= 10
    c5_pass = fatal_errors == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: calibration_gap={probe['calibration_gap']:.4f} <= 0.05"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: z_harm_selectivity_margin={probe['z_harm_selectivity_margin']:.4f} <= 0.01 "
            f"[near={probe['mean_z_harm_norm_near']:.4f} safe={probe['mean_z_harm_norm_safe']:.4f}]"
        )
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: warmup benefit events {warmup_benefit} <= 20")
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: insufficient probes "
            f"(near={probe['n_near_hazard_probes']} safe={probe['n_safe_probes']})"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={fatal_errors}")

    print(f"\nV3-EXQ-015 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "fatal_error_count":           float(fatal_errors),
        "warmup_harm_events":          float(warmup_harm),
        "warmup_benefit_events":       float(warmup_benefit),
        "harm_dim":                    float(HARM_DIM),
        "calibration_gap":             float(probe["calibration_gap"]),
        "mean_net_eval_near_hazard":   float(probe["mean_net_eval_near"]),
        "mean_net_eval_safe":          float(probe["mean_net_eval_safe"]),
        "z_harm_selectivity_margin":   float(probe["z_harm_selectivity_margin"]),
        "mean_z_harm_norm_near":       float(probe["mean_z_harm_norm_near"]),
        "mean_z_harm_norm_safe":       float(probe["mean_z_harm_norm_safe"]),
        "mean_z_world_norm_near":      float(probe["mean_z_world_norm_near"]),
        "mean_z_world_norm_safe":      float(probe["mean_z_world_norm_safe"]),
        "n_near_hazard_probes":        float(probe["n_near_hazard_probes"]),
        "n_safe_probes":               float(probe["n_safe_probes"]),
        "net_eval_pred_std":           float(probe["net_eval_pred_std"]),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-015 — Three-Stream Lateral Head (MECH-099)

**Status:** {status}
**Warmup:** {warmup_episodes} eps (RANDOM policy, 12×12, 15 hazards)
**Probe eval:** {eval_probe_resets} grid resets
**harm_dim:** {HARM_DIM}
**Seed:** {seed}

## Motivation (MECH-099 / SD-007 / SD-003)

EXQ-012 showed calibration_gap ≈ 0.0007 with z_world — E2 identity shortcut.
This experiment tests whether a dedicated lateral encoder head processing ONLY
hazard + contamination channels (lateral stream in MECH-099 biological grounding)
gives E3 a harm-salient z_harm embedding that bypasses the identity shortcut.

Attribution pipeline (lateral variant):
```
z_harm = SplitEncoder.lateral_head(hazard_channels + contamination_view)  # [16-dim]
net_eval: z_harm → scalar ∈ [-1, 1]  (trained on harm_signal values)
causal_signal = net_eval(z_harm_near_hazard) - net_eval(z_harm_safe)
calibration_gap = mean(causal_signal near-hazard) - mean(causal_signal safe)
```

## Probe Results

| Metric | Near-Hazard | Safe | Margin |
|---|---|---|---|
| net_eval (z_harm) | {probe["mean_net_eval_near"]:.4f} | {probe["mean_net_eval_safe"]:.4f} | {probe["calibration_gap"]:.4f} |
| \|\|z_harm\|\| norm | {probe["mean_z_harm_norm_near"]:.4f} | {probe["mean_z_harm_norm_safe"]:.4f} | {probe["z_harm_selectivity_margin"]:.4f} |
| \|\|z_world\|\| norm | {probe["mean_z_world_norm_near"]:.4f} | {probe["mean_z_world_norm_safe"]:.4f} | — |

net_eval pred_std: {probe["net_eval_pred_std"]:.4f}
Warmup: harm={warmup_harm}  benefit={warmup_benefit}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: calibration_gap > 0.05 | {"PASS" if c1_pass else "FAIL"} | {probe["calibration_gap"]:.4f} |
| C2: z_harm selectivity margin > 0.01 | {"PASS" if c2_pass else "FAIL"} | {probe["z_harm_selectivity_margin"]:.4f} |
| C3: Warmup benefit events > 20 | {"PASS" if c3_pass else "FAIL"} | {warmup_benefit} |
| C4: Probe coverage >= 10 each | {"PASS" if c4_pass else "FAIL"} | near={probe["n_near_hazard_probes"]} safe={probe["n_safe_probes"]} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {fatal_errors} |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens"),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": fatal_errors,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--warmup",         type=int,   default=1000)
    parser.add_argument("--probe-resets",   type=int,   default=10)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--harm-scale",     type=float, default=0.02)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_probe_resets=args.probe_resets,
        steps_per_episode=args.steps,
        harm_scale=args.harm_scale,
        alpha_world=args.alpha_world,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
