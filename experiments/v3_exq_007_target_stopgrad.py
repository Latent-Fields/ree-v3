"""
V3-EXQ-007 — SD-003 Self-Attribution: Target Stopgrad Diagnostic

Claim: SD-003

Diagnostic experiment: tests whether encoder co-adaptation causes the
E2 identity shortcut. In r6, both z_start and z_target come from the
same encoder being trained simultaneously. The encoder can learn
representations where consecutive observations map to nearly identical
latents, making E2's job trivially easy (delta≈0 is correct).

Fix: detach z_target in the multi-step E2 loss so gradients only flow
through z_start → E2.world_forward → prediction. The encoder cannot
co-adapt to make E2's job easier — E2 must predict where the *frozen*
target lands.

Same as r6 except _compute_multistep_e2_loss detaches z_target.
Same PASS criteria as r6.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Optional, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_007_target_stopgrad"
CLAIM_IDS = ["SD-003"]

CONDITION_TRAINED = "TRAINED"
CONDITION_RANDOM  = "RANDOM"

RECON_WEIGHT = 1.0
E2_ROLLOUT_STEPS = 5


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _random_cf_action(actual_idx: int, num_actions: int) -> int:
    choices = [a for a in range(num_actions) if a != actual_idx]
    return random.choice(choices) if choices else 0


def _make_world_decoder(world_dim: int, world_obs_dim: int, hidden_dim: int = 64) -> nn.Module:
    return nn.Sequential(
        nn.Linear(world_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, world_obs_dim),
    )


def _compute_multistep_e2_loss(
    agent: REEAgent,
    traj_buffer: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    rollout_steps: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """Multi-step E2 loss WITH TARGET STOPGRAD.

    Key difference from r6: we store raw observations in the traj_buffer
    (not pre-encoded latents). At loss time we re-encode:
      z_start  = world_encoder(obs_start)              # gradients flow
      z_target = world_encoder(obs_target).detach()     # frozen target

    The encoder cannot co-adapt representations to make E2's job easy
    because gradients only flow through z_start, not z_target.
    """
    if len(traj_buffer) < 2:
        return agent.e1.parameters().__next__().new_zeros(())

    encoder = agent.latent_stack.split_encoder.world_encoder

    n = min(batch_size, len(traj_buffer))
    idxs = torch.randperm(len(traj_buffer))[:n].tolist()

    total_loss = agent.e1.parameters().__next__().new_zeros(())
    count = 0
    for idx in idxs:
        segment = traj_buffer[idx]
        if len(segment) < rollout_steps + 1:
            continue

        obs_start = segment[0][0]   # raw obs_world at t
        obs_target = segment[rollout_steps][0]  # raw obs_world at t+N

        # Re-encode: z_start gets gradients, z_target is frozen
        z_start = encoder(obs_start.unsqueeze(0) if obs_start.dim() == 1 else obs_start)
        with torch.no_grad():
            z_target = encoder(obs_target.unsqueeze(0) if obs_target.dim() == 1 else obs_target)

        # Roll forward through E2.world_forward N times
        z = z_start
        for k in range(rollout_steps):
            a_k = segment[k][1]
            z = agent.e2.world_forward(z, a_k)

        total_loss = total_loss + F.mse_loss(z, z_target)
        count += 1

    if count == 0:
        return agent.e1.parameters().__next__().new_zeros(())
    return total_loss / count


def _train_episodes(
    agent: REEAgent,
    env: CausalGridWorld,
    optimizer: optim.Optimizer,
    e3_optimizer: optim.Optimizer,
    world_decoder: nn.Module,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict[str, float]:
    agent.train()
    world_decoder.train()

    # Store raw observations (not latents) so we can re-encode with stopgrad
    traj_buffer: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    MAX_TRAJ_BUFFER = 200

    harm_buffer:    List[torch.Tensor] = []
    no_harm_buffer: List[torch.Tensor] = []

    total_harm_events = 0
    total_recon_loss = 0.0
    total_e2w_loss = 0.0
    recon_count = 0
    e2w_count = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        episode_traj: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            # Store raw obs_world (not latent) for re-encoding with stopgrad
            episode_traj.append((
                obs_world.detach().clone(),
                action.detach(),
            ))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if harm_signal < 0:
                total_harm_events += 1
                harm_buffer.append(latent.z_world.detach())
                if len(harm_buffer) > 500:
                    harm_buffer = harm_buffer[-500:]
            else:
                if step % 3 == 0:
                    no_harm_buffer.append(latent.z_world.detach())
                    if len(no_harm_buffer) > 500:
                        no_harm_buffer = no_harm_buffer[-500:]

            e1_loss      = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()

            e2_world_loss = _compute_multistep_e2_loss(
                agent, traj_buffer, E2_ROLLOUT_STEPS, batch_size=8
            )
            if e2_world_loss.item() > 0:
                total_e2w_loss += e2_world_loss.item()
                e2w_count += 1

            obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
            z_world_for_recon = agent.latent_stack.split_encoder.world_encoder(obs_w)
            recon = world_decoder(z_world_for_recon)
            recon_loss = F.mse_loss(recon, obs_w)
            total_recon_loss += recon_loss.item()
            recon_count += 1

            e12_loss = e1_loss + e2_self_loss + e2_world_loss + RECON_WEIGHT * recon_loss
            if e12_loss.requires_grad:
                optimizer.zero_grad()
                e12_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(world_decoder.parameters(), 1.0)
                optimizer.step()

            n_h  = len(harm_buffer)
            n_nh = len(no_harm_buffer)
            if n_h >= 4 and n_nh >= 4:
                k = min(16, n_h, n_nh)
                zw_harm    = torch.cat([harm_buffer[i]    for i in torch.randperm(n_h)[:k].tolist()], dim=0)
                zw_no_harm = torch.cat([no_harm_buffer[i] for i in torch.randperm(n_nh)[:k].tolist()], dim=0)
                zw_batch = torch.cat([zw_harm, zw_no_harm], dim=0)
                labels_t = torch.cat([
                    torch.ones(k,  1, device=agent.device),
                    torch.zeros(k, 1, device=agent.device),
                ], dim=0)
                harm_pred = agent.e3.harm_eval(zw_batch)
                if not torch.isnan(harm_pred).any():
                    e3_loss = F.binary_cross_entropy(
                        harm_pred.clamp(1e-6, 1 - 1e-6), labels_t
                    )
                    e3_optimizer.zero_grad()
                    e3_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    e3_optimizer.step()

            if done:
                break

        for start in range(0, len(episode_traj) - E2_ROLLOUT_STEPS):
            traj_buffer.append(episode_traj[start:start + E2_ROLLOUT_STEPS + 1])
        if len(traj_buffer) > MAX_TRAJ_BUFFER:
            traj_buffer = traj_buffer[-MAX_TRAJ_BUFFER:]

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            avg_recon = total_recon_loss / max(1, recon_count)
            avg_e2w = total_e2w_loss / max(1, e2w_count)
            print(f"  [train] ep {ep + 1}/{num_episodes}  "
                  f"harm={total_harm_events}  "
                  f"recon_loss={avg_recon:.5f}  "
                  f"e2w_loss={avg_e2w:.6f}  "
                  f"traj_segs={len(traj_buffer)}  "
                  f"harm_buf={len(harm_buffer)}  no_harm_buf={len(no_harm_buffer)}",
                  flush=True)

    return {
        "total_harm_events": total_harm_events,
        "mean_recon_loss": total_recon_loss / max(1, recon_count),
        "mean_e2w_loss": total_e2w_loss / max(1, e2w_count),
    }


def _eval_probes(
    agent: REEAgent,
    env: CausalGridWorld,
    num_resets: int,
    condition_label: str,
) -> Dict:
    agent.eval()
    near_sigs: List[float] = []
    safe_sigs:  List[float] = []
    fatal_errors = 0

    wall_type   = env.ENTITY_TYPES["wall"]
    hazard_type = env.ENTITY_TYPES["hazard"]

    def _run_probe(ax: int, ay: int, actual_idx: int) -> float:
        env.agent_x = ax
        env.agent_y = ay
        obs_dict = env._get_observation_dict()
        with torch.no_grad():
            latent  = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            z_world = latent.z_world
            cf_idx  = _random_cf_action(actual_idx, env.action_dim)
            a_act = _action_to_onehot(actual_idx, env.action_dim, agent.device)
            a_cf  = _action_to_onehot(cf_idx,     env.action_dim, agent.device)
            zw_act = agent.e2.world_forward(z_world, a_act)
            zw_cf  = agent.e2.world_forward(z_world, a_cf)
            h_act = torch.nan_to_num(agent.e3.harm_eval(zw_act), nan=0.5)
            h_cf  = torch.nan_to_num(agent.e3.harm_eval(zw_cf),  nan=0.5)
            return float((h_act - h_cf).item())

    try:
        for _ in range(num_resets):
            env.reset()
            for hx, hy in env.hazards:
                for action_idx, (dx, dy) in env.ACTIONS.items():
                    if action_idx == 4:
                        continue
                    ax, ay = hx - dx, hy - dy
                    if 0 <= ax < env.size and 0 <= ay < env.size:
                        cell = int(env.grid[ax, ay])
                        if cell not in (wall_type, hazard_type):
                            near_sigs.append(_run_probe(ax, ay, action_idx))

            for px in range(env.size):
                for py in range(env.size):
                    if int(env.grid[px, py]) in (wall_type, hazard_type):
                        continue
                    min_dist = min(abs(px - hx) + abs(py - hy) for hx, hy in env.hazards)
                    if min_dist > 3:
                        safe_sigs.append(_run_probe(px, py, random.randint(0, 3)))

    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL [{condition_label}]: {traceback.format_exc()}", flush=True)

    mean_near = float(sum(near_sigs) / max(1, len(near_sigs)))
    mean_safe = float(sum(safe_sigs)  / max(1, len(safe_sigs)))
    calibration_gap = mean_near - mean_safe
    harm_eval_degenerate = (abs(mean_near) < 1e-6 and abs(mean_safe) < 1e-6)

    print(f"  [{condition_label}] "
          f"n_near={len(near_sigs)} n_safe={len(safe_sigs)}  "
          f"mean_near={mean_near:.4f} mean_safe={mean_safe:.4f}  "
          f"gap={calibration_gap:.4f}  "
          f"{'[DEGENERATE]' if harm_eval_degenerate else ''}",
          flush=True)

    return {
        "condition": condition_label,
        "calibration_gap": calibration_gap,
        "mean_causal_sig_near_hazard": mean_near,
        "mean_causal_sig_safe": mean_safe,
        "n_near_hazard_probes": len(near_sigs),
        "n_safe_probes": len(safe_sigs),
        "harm_eval_degenerate": harm_eval_degenerate,
        "fatal_errors": fatal_errors,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 300,
    eval_probe_resets: int = 10,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(seed=seed, num_hazards=8)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
    )

    fatal_errors = 0
    results_by_condition = {}

    print(f"\n[V3-EXQ-007] Seed {seed} Condition TRAINED (target stopgrad)", flush=True)
    agent_trained = REEAgent(config)
    world_decoder = _make_world_decoder(world_dim, env.world_obs_dim)

    e12_params = [p for n, p in agent_trained.named_parameters() if "harm_eval" not in n]
    e12_params += list(world_decoder.parameters())
    opt    = optim.Adam(e12_params, lr=lr)
    e3_opt = optim.Adam(agent_trained.e3.harm_eval_head.parameters(), lr=1e-4)

    print(f"  Warmup: {warmup_episodes} episodes (RANDOM policy + recon + stopgrad targets) ...", flush=True)
    train_metrics = _train_episodes(
        agent_trained, env, opt, e3_opt, world_decoder, warmup_episodes, steps_per_episode
    )
    warmup_harm = train_metrics["total_harm_events"]
    mean_recon  = train_metrics["mean_recon_loss"]
    mean_e2w    = train_metrics["mean_e2w_loss"]
    print(f"  Warmup complete. Harm={warmup_harm}  mean_recon={mean_recon:.5f}  mean_e2w={mean_e2w:.6f}", flush=True)

    print(f"  Probe eval ({eval_probe_resets} grid resets) ...", flush=True)
    r_trained = _eval_probes(agent_trained, env, eval_probe_resets, CONDITION_TRAINED)
    results_by_condition[CONDITION_TRAINED] = r_trained
    fatal_errors += r_trained["fatal_errors"]

    print(f"\n[V3-EXQ-007] Seed {seed} Condition RANDOM", flush=True)
    torch.manual_seed(seed + 5000)
    random.seed(seed + 5000)
    agent_random = REEAgent(config)
    print(f"  Probe eval ({eval_probe_resets} grid resets, no training) ...", flush=True)
    r_random = _eval_probes(agent_random, env, eval_probe_resets, CONDITION_RANDOM)
    results_by_condition[CONDITION_RANDOM] = r_random
    fatal_errors += r_random["fatal_errors"]

    trained_gap = results_by_condition[CONDITION_TRAINED]["calibration_gap"]
    random_gap  = results_by_condition[CONDITION_RANDOM]["calibration_gap"]
    n_near      = results_by_condition[CONDITION_TRAINED]["n_near_hazard_probes"]
    n_safe      = results_by_condition[CONDITION_TRAINED]["n_safe_probes"]
    trained_deg = results_by_condition[CONDITION_TRAINED]["harm_eval_degenerate"]

    crit1_pass = trained_gap > 0.05
    crit2_pass = abs(random_gap) < 0.10
    crit3_pass = warmup_harm > 100
    crit4_pass = fatal_errors == 0
    crit5_pass = not trained_deg
    crit6_pass = n_near >= 10 and n_safe >= 10

    all_pass = all([crit1_pass, crit2_pass, crit3_pass, crit4_pass, crit5_pass, crit6_pass])
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([crit1_pass, crit2_pass, crit3_pass, crit4_pass, crit5_pass, crit6_pass])

    failure_notes = []
    if not crit1_pass: failure_notes.append(f"C1 FAIL: TRAINED calibration_gap {trained_gap:.4f} <= 0.05")
    if not crit2_pass: failure_notes.append(f"C2 FAIL: RANDOM |gap| {abs(random_gap):.4f} >= 0.10")
    if not crit3_pass: failure_notes.append(f"C3 FAIL: warmup harm events {warmup_harm} <= 100")
    if not crit4_pass: failure_notes.append(f"C4 FAIL: fatal_errors={fatal_errors}")
    if not crit5_pass: failure_notes.append("C5 FAIL: harm_eval collapsed to constant")
    if not crit6_pass: failure_notes.append(f"C6 FAIL: insufficient probes (near={n_near} safe={n_safe})")

    print(f"\nSD-003 / V3-EXQ-007 verdict: {status}  ({criteria_met}/6)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    t = results_by_condition[CONDITION_TRAINED]
    r = results_by_condition[CONDITION_RANDOM]

    metrics = {
        "fatal_error_count": float(fatal_errors),
        "warmup_harm_events": float(warmup_harm),
        "mean_recon_loss": float(mean_recon),
        "mean_e2_world_loss": float(mean_e2w),
        "e2_rollout_steps": float(E2_ROLLOUT_STEPS),
        "target_stopgrad": 1.0,
        "trained_calibration_gap": float(trained_gap),
        "random_calibration_gap": float(random_gap),
        "trained_mean_near_hazard": float(t["mean_causal_sig_near_hazard"]),
        "trained_mean_safe":        float(t["mean_causal_sig_safe"]),
        "random_mean_near_hazard":  float(r["mean_causal_sig_near_hazard"]),
        "random_mean_safe":         float(r["mean_causal_sig_safe"]),
        "n_near_hazard_probes": float(n_near),
        "n_safe_probes":        float(n_safe),
        "trained_harm_eval_degenerate": 1.0 if trained_deg else 0.0,
        "crit1_pass": 1.0 if crit1_pass else 0.0,
        "crit2_pass": 1.0 if crit2_pass else 0.0,
        "crit3_pass": 1.0 if crit3_pass else 0.0,
        "crit4_pass": 1.0 if crit4_pass else 0.0,
        "crit5_pass": 1.0 if crit5_pass else 0.0,
        "crit6_pass": 1.0 if crit6_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-007 — SD-003 Target Stopgrad Diagnostic

**Status:** {status}
**Warmup:** {warmup_episodes} episodes, RANDOM policy + recon loss + {E2_ROLLOUT_STEPS}-step E2 rollouts
**Design change:** z_target detached in multi-step E2 loss (prevents encoder co-adaptation)
**Probe eval:** {eval_probe_resets} grid resets x (near-hazard + safe positions)
**Seed:** {seed}

## Hypothesis

r6 E2 identity shortcut may be caused by encoder co-adaptation: both
z_start and z_target come from the same encoder trained in the same
backward pass. The encoder could learn representations where consecutive
observations map to nearly identical latents, making delta≈0 correct.

Detaching z_target prevents this — E2 must predict where the *frozen*
target lands. If this fixes the gap, encoder collapse is the bottleneck.
If not, the grid environment itself lacks sufficient per-step variation.

Mean E2 world loss: {mean_e2w:.6f} | Mean reconstruction loss: {mean_recon:.5f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: TRAINED calibration_gap > 0.05 | {"PASS" if crit1_pass else "FAIL"} | {trained_gap:.4f} |
| C2: RANDOM abs(calibration_gap) < 0.10 | {"PASS" if crit2_pass else "FAIL"} | {abs(random_gap):.4f} |
| C3: Warmup harm events > 100 | {"PASS" if crit3_pass else "FAIL"} | {warmup_harm} |
| C4: No fatal errors | {"PASS" if crit4_pass else "FAIL"} | {fatal_errors} |
| C5: harm_eval non-degenerate | {"PASS" if crit5_pass else "FAIL"} | {"OK" if not trained_deg else "COLLAPSED"} |
| C6: Probe coverage >= 10 each | {"PASS" if crit6_pass else "FAIL"} | near={n_near} safe={n_safe} |

## Calibration Results (Probe-Based)

| Condition | mean_causal_sig(near_hazard) | mean_causal_sig(safe) | calibration_gap |
|---|---|---|---|
| TRAINED | {t["mean_causal_sig_near_hazard"]:.4f} | {t["mean_causal_sig_safe"]:.4f} | {trained_gap:.4f} |
| RANDOM  | {r["mean_causal_sig_near_hazard"]:.4f} | {r["mean_causal_sig_safe"]:.4f} | {random_gap:.4f} |

Criteria met: {criteria_met}/6 -> **{status}**
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
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--warmup",       type=int, default=300)
    parser.add_argument("--probe-resets", type=int, default=10)
    parser.add_argument("--steps",        type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_probe_resets=args.probe_resets,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0] if CLAIM_IDS else EXPERIMENT_TYPE
    result["verdict"] = result["status"]

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
