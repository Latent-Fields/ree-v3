"""
V3-EXQ-012 — SD-003 Signed Net Eval (Benefit + Harm Signal)

Claim: SD-003 (self-attribution via counterfactual E2+E3 joint pipeline).

Hypothesis (2026-03-17):
  EXQ-002 through EXQ-010 plateau at calibration_gap ≈ 0.027 because E3's
  harm_eval is a binary classifier (harm=1, no-harm=0). It cannot distinguish
  beneficial z_world states from neutral ones — both labelled 0. This discards
  half the available training signal.

  CausalGridWorld harm_signal is already SIGNED:
    agent_caused_hazard: -0.4   env_caused_hazard: -0.5
    resource:            +0.3   none:               0.0

  Fix: replace binary BCE with a REGRESSION net_eval head trained on actual
  harm_signal values. E3 now learns the full ±0.3–0.5 value boundary.

  The causal signature:
    causal_sig = net_eval(E2.world_forward(z_world, a_actual))
               - net_eval(E2.world_forward(z_world, a_cf))
  has sharper contrast because:
  - z_world_actual (entering hazard) → net_eval ≈ -0.4 to -0.5
  - z_world_cf (safe or resource cell) → net_eval ≈ 0.0 to +0.3
  - Difference is 0.4–0.8 vs ~0.1 under binary classification

All other components identical to EXQ-002r6:
  - Multi-step E2 training (N=5 rollouts)
  - Reconstruction loss on world encoder
  - RANDOM exploration policy during warmup
  - Probe-based eval (curated near-hazard / safe positions)
  - Separate E3 optimizer

PASS criteria (ALL must hold):
  C1: TRAINED calibration_gap > 0.05
  C2: RANDOM  |calibration_gap| < 0.10
  C3: warmup harm events > 100
  C4: fatal_error_count == 0
  C5: net_eval non-degenerate (std > 0.01 over eval probes)
  C6: n_near_hazard_probes >= 10 AND n_safe_probes >= 10
  C7: warmup benefit events > 20  (resources collected — confirms benefit signal seen)
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


EXPERIMENT_TYPE = "v3_exq_012_sd003_net_eval"
CLAIM_IDS = ["SD-003"]

CONDITION_TRAINED = "TRAINED"
CONDITION_RANDOM  = "RANDOM"

RECON_WEIGHT   = 1.0
E2_ROLLOUT_STEPS = 5

# net_eval regression buffer: store (z_world, signal_value) pairs
# signal_value is the raw harm_signal from the environment (signed float)


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


def _make_net_eval_head(world_dim: int, hidden_dim: int = 64) -> nn.Module:
    """
    Signed regression head: z_world → scalar net value ∈ [-1, 1].
    Replaces binary harm_eval BCE with regression on actual harm_signal values.
    Output passes through tanh to keep in [-1, 1] range.
    """
    return nn.Sequential(
        nn.Linear(world_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
        nn.Tanh(),
    )


def _compute_multistep_e2_loss(
    agent: REEAgent,
    traj_buffer: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    rollout_steps: int,
    batch_size: int = 8,
) -> torch.Tensor:
    if len(traj_buffer) < 2:
        return next(agent.e1.parameters()).sum() * 0.0
    n = min(batch_size, len(traj_buffer))
    idxs = torch.randperm(len(traj_buffer))[:n].tolist()
    total_loss = next(agent.e1.parameters()).sum() * 0.0
    count = 0
    for idx in idxs:
        segment = traj_buffer[idx]
        if len(segment) < rollout_steps + 1:
            continue
        z_start  = segment[0][0]
        z_target = segment[rollout_steps][0]
        z = z_start
        for k in range(rollout_steps):
            a_k = segment[k][1]
            z = agent.e2.world_forward(z, a_k)
        total_loss = total_loss + F.mse_loss(z, z_target)
        count += 1
    return total_loss / count if count > 0 else total_loss


def _train_episodes(
    agent: REEAgent,
    env: CausalGridWorld,
    net_eval_head: nn.Module,
    optimizer: optim.Optimizer,
    net_eval_optimizer: optim.Optimizer,
    world_decoder: nn.Module,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    agent.train()
    world_decoder.train()
    net_eval_head.train()

    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200

    # Regression buffer: (z_world, signal_value) for all non-zero signal steps
    # Also include a sample of zero-signal steps for calibration
    signal_buffer: List[Tuple[torch.Tensor, float]] = []
    MAX_SIGNAL_BUFFER = 1000

    total_harm_events   = 0
    total_benefit_events = 0
    total_recon_loss    = 0.0
    total_e2w_loss      = 0.0
    recon_count = 0
    e2w_count   = 0

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
                total_harm_events += 1
            elif harm_signal > 0:
                total_benefit_events += 1

            # Always record this step's (z_world, signal) for net_eval regression
            # Include ALL non-zero steps; sample zero steps at ~1:3 ratio
            if harm_signal != 0.0 or (step % 3 == 0):
                signal_buffer.append((latent.z_world.detach(), float(harm_signal)))
                if len(signal_buffer) > MAX_SIGNAL_BUFFER:
                    signal_buffer = signal_buffer[-MAX_SIGNAL_BUFFER:]

            # ── E1 + E2w + recon backward pass ──
            e1_loss = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()

            e2_world_loss = _compute_multistep_e2_loss(
                agent, traj_buffer, E2_ROLLOUT_STEPS, batch_size=8
            )
            if e2_world_loss.item() > 0:
                total_e2w_loss += e2_world_loss.item()
                e2w_count += 1

            obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
            z_world_recon = agent.latent_stack.split_encoder.world_encoder(obs_w)
            recon = world_decoder(z_world_recon)
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

            # ── net_eval regression backward pass ──
            if len(signal_buffer) >= 8:
                k = min(32, len(signal_buffer))
                idxs = torch.randperm(len(signal_buffer))[:k].tolist()
                zw_batch = torch.cat([signal_buffer[i][0] for i in idxs], dim=0)
                sv_batch = torch.tensor(
                    [signal_buffer[i][1] for i in idxs],
                    device=agent.device,
                ).unsqueeze(1)  # [k, 1]
                pred = net_eval_head(zw_batch)
                # Normalise targets to [-1, 1] (max signal magnitude is 0.5)
                sv_norm = sv_batch / 0.5
                sv_norm = sv_norm.clamp(-1.0, 1.0)
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

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            avg_recon = total_recon_loss / max(1, recon_count)
            avg_e2w   = total_e2w_loss   / max(1, e2w_count)
            print(f"  [train] ep {ep+1}/{num_episodes}  "
                  f"harm={total_harm_events}  benefit={total_benefit_events}  "
                  f"recon={avg_recon:.5f}  e2w={avg_e2w:.6f}  "
                  f"sig_buf={len(signal_buffer)}",
                  flush=True)

    return {
        "total_harm_events":    total_harm_events,
        "total_benefit_events": total_benefit_events,
        "mean_recon_loss":      total_recon_loss / max(1, recon_count),
        "mean_e2w_loss":        total_e2w_loss   / max(1, e2w_count),
    }


def _eval_probes(
    agent: REEAgent,
    net_eval_head: nn.Module,
    env: CausalGridWorld,
    num_resets: int,
    condition_label: str,
) -> Dict:
    agent.eval()
    net_eval_head.eval()
    near_sigs: List[float] = []
    safe_sigs:  List[float] = []
    fatal_errors = 0
    all_pred_vals: List[float] = []

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
            v_act = net_eval_head(zw_act)
            v_cf  = net_eval_head(zw_cf)
            all_pred_vals.extend([float(v_act.item()), float(v_cf.item())])
            return float((v_act - v_cf).item())

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

    pred_std = float(torch.tensor(all_pred_vals).std().item()) if len(all_pred_vals) > 1 else 0.0
    degenerate = pred_std < 0.01

    print(f"  [{condition_label}] "
          f"n_near={len(near_sigs)} n_safe={len(safe_sigs)}  "
          f"mean_near={mean_near:.4f} mean_safe={mean_safe:.4f}  "
          f"gap={calibration_gap:.4f}  pred_std={pred_std:.4f}  "
          f"{'[DEGENERATE]' if degenerate else ''}",
          flush=True)

    return {
        "condition": condition_label,
        "calibration_gap": calibration_gap,
        "mean_causal_sig_near_hazard": mean_near,
        "mean_causal_sig_safe": mean_safe,
        "n_near_hazard_probes": len(near_sigs),
        "n_safe_probes": len(safe_sigs),
        "net_eval_pred_std": pred_std,
        "net_eval_degenerate": degenerate,
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

    env = CausalGridWorld(seed=seed, num_hazards=8, num_resources=5)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
    )

    fatal_errors = 0
    results_by_condition = {}

    # ── TRAINED ──────────────────────────────────────────────────────────
    print(f"\n[V3-EXQ-012] Seed {seed} Condition TRAINED", flush=True)
    agent_trained   = REEAgent(config)
    world_decoder   = _make_world_decoder(world_dim, env.world_obs_dim)
    net_eval_head   = _make_net_eval_head(world_dim)

    # E12 optimizer excludes harm_eval (we're replacing it with net_eval_head)
    e12_params = [p for n, p in agent_trained.named_parameters() if "harm_eval" not in n]
    e12_params += list(world_decoder.parameters())
    opt         = optim.Adam(e12_params, lr=lr)
    net_eval_opt = optim.Adam(net_eval_head.parameters(), lr=1e-4)

    print(f"  Warmup: {warmup_episodes} eps (RANDOM + recon + {E2_ROLLOUT_STEPS}-step E2 + net_eval regression)",
          flush=True)
    train_metrics = _train_episodes(
        agent_trained, env, net_eval_head, opt, net_eval_opt, world_decoder,
        warmup_episodes, steps_per_episode,
    )
    warmup_harm    = train_metrics["total_harm_events"]
    warmup_benefit = train_metrics["total_benefit_events"]
    mean_recon     = train_metrics["mean_recon_loss"]
    mean_e2w       = train_metrics["mean_e2w_loss"]
    print(f"  Warmup done. harm={warmup_harm}  benefit={warmup_benefit}  "
          f"recon={mean_recon:.5f}  e2w={mean_e2w:.6f}", flush=True)

    r_trained = _eval_probes(agent_trained, net_eval_head, env, eval_probe_resets, CONDITION_TRAINED)
    results_by_condition[CONDITION_TRAINED] = r_trained
    fatal_errors += r_trained["fatal_errors"]

    # ── RANDOM ───────────────────────────────────────────────────────────
    print(f"\n[V3-EXQ-012] Seed {seed} Condition RANDOM", flush=True)
    torch.manual_seed(seed + 5000)
    random.seed(seed + 5000)
    agent_random     = REEAgent(config)
    net_eval_random  = _make_net_eval_head(world_dim)  # untrained
    r_random = _eval_probes(agent_random, net_eval_random, env, eval_probe_resets, CONDITION_RANDOM)
    results_by_condition[CONDITION_RANDOM] = r_random
    fatal_errors += r_random["fatal_errors"]

    # ── PASS / FAIL ───────────────────────────────────────────────────────
    trained_gap  = r_trained["calibration_gap"]
    random_gap   = r_random["calibration_gap"]
    n_near       = r_trained["n_near_hazard_probes"]
    n_safe       = r_trained["n_safe_probes"]
    trained_deg  = r_trained["net_eval_degenerate"]

    c1 = trained_gap > 0.05
    c2 = abs(random_gap) < 0.10
    c3 = warmup_harm > 100
    c4 = fatal_errors == 0
    c5 = not trained_deg
    c6 = n_near >= 10 and n_safe >= 10
    c7 = warmup_benefit > 20

    all_pass = all([c1, c2, c3, c4, c5, c6, c7])
    status   = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1, c2, c3, c4, c5, c6, c7])

    failure_notes = []
    if not c1: failure_notes.append(f"C1 FAIL: TRAINED gap {trained_gap:.4f} <= 0.05")
    if not c2: failure_notes.append(f"C2 FAIL: RANDOM |gap| {abs(random_gap):.4f} >= 0.10")
    if not c3: failure_notes.append(f"C3 FAIL: warmup harm {warmup_harm} <= 100")
    if not c4: failure_notes.append(f"C4 FAIL: fatal_errors={fatal_errors}")
    if not c5: failure_notes.append("C5 FAIL: net_eval collapsed (pred_std < 0.01)")
    if not c6: failure_notes.append(f"C6 FAIL: insufficient probes (near={n_near} safe={n_safe})")
    if not c7: failure_notes.append(f"C7 FAIL: warmup benefit events {warmup_benefit} <= 20 (no benefit signal seen)")

    print(f"\nV3-EXQ-012 verdict: {status}  ({criteria_met}/7)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    t = r_trained
    r = r_random

    metrics = {
        "fatal_error_count":       float(fatal_errors),
        "warmup_harm_events":      float(warmup_harm),
        "warmup_benefit_events":   float(warmup_benefit),
        "mean_recon_loss":         float(mean_recon),
        "mean_e2_world_loss":      float(mean_e2w),
        "e2_rollout_steps":        float(E2_ROLLOUT_STEPS),
        "trained_calibration_gap": float(trained_gap),
        "random_calibration_gap":  float(random_gap),
        "trained_mean_near_hazard": float(t["mean_causal_sig_near_hazard"]),
        "trained_mean_safe":        float(t["mean_causal_sig_safe"]),
        "random_mean_near_hazard":  float(r["mean_causal_sig_near_hazard"]),
        "random_mean_safe":         float(r["mean_causal_sig_safe"]),
        "n_near_hazard_probes":    float(n_near),
        "n_safe_probes":           float(n_safe),
        "trained_net_eval_std":    float(t["net_eval_pred_std"]),
        "trained_net_eval_degenerate": 1.0 if trained_deg else 0.0,
        "crit1_pass": 1.0 if c1 else 0.0,
        "crit2_pass": 1.0 if c2 else 0.0,
        "crit3_pass": 1.0 if c3 else 0.0,
        "crit4_pass": 1.0 if c4 else 0.0,
        "crit5_pass": 1.0 if c5 else 0.0,
        "crit6_pass": 1.0 if c6 else 0.0,
        "crit7_pass": 1.0 if c7 else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-012 — SD-003 Signed Net Eval (Benefit + Harm Signal)

**Status:** {status}
**Warmup:** {warmup_episodes} eps, RANDOM policy + recon loss + {E2_ROLLOUT_STEPS}-step E2 + net_eval regression
**Probe eval:** {eval_probe_resets} grid resets × (near-hazard + safe positions)
**Seed:** {seed}

## Key Change vs EXQ-002 through EXQ-010

Binary BCE `harm_eval` (harm=1, no-harm=0) replaced with signed REGRESSION
`net_eval` trained on actual `harm_signal` values (normalised to [-1, 1]).

- `agent_caused_hazard` → target ≈ -0.8  (`-0.4 / 0.5`)
- `env_caused_hazard`   → target ≈ -1.0  (`-0.5 / 0.5`)
- `resource`            → target ≈ +0.6  (`+0.3 / 0.5`)
- `none`                → target ≈  0.0

E3 now has full ±0.3–0.5 value boundaries. Benefit/harm contrast amplifies
the SD-003 causal signature.

Mean E2 world loss: {mean_e2w:.6f}  |  Mean reconstruction loss: {mean_recon:.5f}
Warmup harm events: {warmup_harm}   |  Warmup benefit events: {warmup_benefit}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: TRAINED calibration_gap > 0.05 | {"PASS" if c1 else "FAIL"} | {trained_gap:.4f} |
| C2: RANDOM abs(calibration_gap) < 0.10 | {"PASS" if c2 else "FAIL"} | {abs(random_gap):.4f} |
| C3: Warmup harm events > 100 | {"PASS" if c3 else "FAIL"} | {warmup_harm} |
| C4: No fatal errors | {"PASS" if c4 else "FAIL"} | {fatal_errors} |
| C5: net_eval non-degenerate | {"PASS" if c5 else "FAIL"} | std={t["net_eval_pred_std"]:.4f} |
| C6: Probe coverage >= 10 each | {"PASS" if c6 else "FAIL"} | near={n_near} safe={n_safe} |
| C7: Warmup benefit events > 20 | {"PASS" if c7 else "FAIL"} | {warmup_benefit} |

## Calibration Results

| Condition | mean_causal_sig(near_hazard) | mean_causal_sig(safe) | calibration_gap |
|---|---|---|---|
| TRAINED | {t["mean_causal_sig_near_hazard"]:.4f} | {t["mean_causal_sig_safe"]:.4f} | {trained_gap:.4f} |
| RANDOM  | {r["mean_causal_sig_near_hazard"]:.4f} | {r["mean_causal_sig_safe"]:.4f} | {random_gap:.4f} |

## Attribution Pipeline

```
z_world = encoder(obs_world)                    [reconstruction-loss trained]
z_world_actual = E2.world_forward(z_world, a_actual)
z_world_cf     = E2.world_forward(z_world, a_cf)
v_actual = net_eval(z_world_actual)             [SIGNED regression, not binary]
v_cf     = net_eval(z_world_cf)
causal_sig = v_actual - v_cf
```

Criteria met: {criteria_met}/7 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if criteria_met >= 4 else "weakens"),
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
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
