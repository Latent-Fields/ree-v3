#!/opt/local/bin/python3
"""
V3-EXQ-066 -- MECH-089: Theta-Batching Reduces E3 Prediction Error

Claim: MECH-089
  "E1 (fast sensorium) updates are theta-batched before E3 samples them.
   E3 sees theta-cycle-averaged E1 summaries, not raw step-by-step updates."

Motivation:
  MECH-089 predicts that temporal averaging of z_world in the ThetaBuffer
  filters within-theta-cycle noise before E3 samples the world state. If this
  filtering is genuine (and not just smoothing out meaningful signal), then:
    (1) E3 prediction error should be lower in the batched condition.
    (2) E1 z_world updates should exhibit measurable within-batch variance,
        confirming that there is signal to average (not a trivially flat series).
    (3) E3 should not collapse in either condition (harm_pred_std > 0.01).

  SD-006 implements multi-rate execution. The ThetaBuffer (theta_buffer.py)
  is the cross-rate integration mechanism: E1 pushes z_world each step;
  E3 calls summary() at its tick, receiving the mean over the last
  theta_buffer_size entries.

  This experiment directly tests whether that averaging reduces E3's noise
  exposure: condition A (k=4 batched) vs condition B (k=1 raw, every step).

Design:
  Two conditions, matched random seeds:
    Condition A (batched):
      theta_buffer_size = 4   (E3 sees average of last 4 E1 z_world estimates)
      e3_steps_per_tick = 4   (E3 samples once per k=4 steps)
      E1 runs every step (e1_steps_per_tick = 1)
    Condition B (raw):
      theta_buffer_size = 1   (E3 sees only the current step's z_world -- no averaging)
      e3_steps_per_tick = 1   (E3 samples every step)
      E1 runs every step (e1_steps_per_tick = 1)

  k = 4 is used as the theta-to-gamma ratio proxy.

  Metric: E3 prediction error = MSE between E3's harm_eval output at step t
  and the harm signal received at step t. Variance over the episode is the
  noise measure.

  Within-batch variance (batch_uniformity): variance of z_world entries
  within each batch window -- confirms batching is averaging non-trivial content.

PASS criteria (ALL must hold; pre-registered before running):
  C1: e3_prediction_error_batched < e3_prediction_error_raw * 0.90
      -- batching reduces E3 harm-eval noise by at least 10%
  C2: e1_updates_per_e3_batched >= k - 1
      -- E1 actually ran multiple times per E3 sample (batching happening)
  C3: harm_pred_std_batched > 0.01
      -- E3 harm evaluator not collapsed in batched condition
  C4: harm_pred_std_raw > 0.01
      -- E3 harm evaluator not collapsed in raw condition
  C5: No fatal errors in either condition
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_066_mech089_theta_batching"
CLAIM_IDS = ["MECH-089"]
THETA_K = 4  # theta-to-gamma ratio proxy


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _var_safe(lst: List[float]) -> float:
    """Population variance of a list."""
    if len(lst) < 2:
        return 0.0
    mu = _mean_safe(lst)
    return float(sum((x - mu) ** 2 for x in lst) / len(lst))


def _build_config(
    env: CausalGridWorldV2,
    theta_buffer_size: int,
    e3_steps_per_tick: int,
    alpha_world: float,
    self_dim: int,
    world_dim: int,
) -> REEConfig:
    """Build a REEConfig with specified theta buffer and E3 tick rate."""
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
    )
    # Override heartbeat params directly -- from_dims does not expose these
    config.heartbeat.theta_buffer_size = theta_buffer_size
    config.heartbeat.e3_steps_per_tick = e3_steps_per_tick
    config.heartbeat.e2_steps_per_tick = 1   # E2 every step (keep E2 identical across conditions)
    config.heartbeat.e1_steps_per_tick = 1   # E1 every step (identical)
    # Disable MECH-093 beta-modulated rate to keep conditions clean
    config.heartbeat.beta_rate_min_steps = e3_steps_per_tick
    config.heartbeat.beta_rate_max_steps = e3_steps_per_tick
    return config


def _run_condition(
    env: CausalGridWorldV2,
    seed: int,
    theta_buffer_size: int,
    e3_steps_per_tick: int,
    num_episodes: int,
    steps_per_episode: int,
    alpha_world: float,
    lr: float,
    self_dim: int,
    world_dim: int,
    label: str,
) -> Dict:
    """
    Run one condition (batched or raw) with fresh agent and matched seed.

    Returns metrics dict.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    # env is created fresh for each condition with the same seed -- no .seed() method on CausalGridWorldV2

    config = _build_config(env, theta_buffer_size, e3_steps_per_tick, alpha_world, self_dim, world_dim)
    agent = REEAgent(config)

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=lr,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    harm_pred_errors: List[float] = []   # |harm_pred - harm_signal|^2 per step
    harm_preds_all: List[float] = []     # harm_eval outputs (for std)
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    within_batch_vars: List[float] = []  # variance of z_world within each E3 window
    e1_count = 0
    e3_count = 0
    fatal = 0

    # Track z_world within the current E3 window for within-batch variance
    current_window_z: List[torch.Tensor] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        current_window_z.clear()

        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            # Accumulate within-window z_world for batch uniformity measure
            current_window_z.append(z_world_curr.clone())

            if ticks.get("e1_tick", False):
                e1_count += 1

            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()

            if ticks.get("e3_tick", False) and candidates:
                e3_count += 1

                # Record within-batch z_world variance before E3 samples
                if len(current_window_z) >= 2:
                    window_stack = torch.stack(current_window_z, dim=0)  # [T, 1, world_dim]
                    window_var = float(window_stack.var(dim=0).mean().item())
                    within_batch_vars.append(window_var)
                current_window_z.clear()

                try:
                    action = agent.select_action(candidates, ticks, temperature=1.0)
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action
                except Exception:
                    fatal += 1
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action
            else:
                action = agent._last_action
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # E3 prediction error: compare harm_eval(theta_z) to harm signal
            try:
                with torch.no_grad():
                    harm_pred = float(agent.e3.harm_eval(theta_z).item())
                harm_preds_all.append(harm_pred)
                # harm_signal < 0 means hazard; map to [0, 1] target for error
                harm_target = 1.0 if harm_signal < 0 else 0.0
                harm_pred_errors.append((harm_pred - harm_target) ** 2)
            except Exception:
                pass

            # World-forward buffer
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            # Harm evaluation training buffers
            if harm_signal < 0:
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

            # E1 loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            # World-forward (E2) loss
            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_optimizer.step()

            # Harm evaluator training
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat([
                    torch.ones(k_p, 1, device=agent.device),
                    torch.zeros(k_n, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                current_window_z.clear()
                break

    e3_pred_error_var = _var_safe(harm_pred_errors)
    harm_pred_std = float(torch.tensor(harm_preds_all).std().item()) if len(harm_preds_all) > 1 else 0.0
    batch_uniformity = _mean_safe(within_batch_vars)  # mean within-batch z_world variance
    e1_per_e3 = (e1_count / max(1, e3_count)) - 1.0   # expected k-1

    print(
        f"  [{label}] e1={e1_count}  e3={e3_count}"
        f"  e1_per_e3={e1_per_e3:.2f}  e3_pred_err_var={e3_pred_error_var:.5f}"
        f"  harm_pred_std={harm_pred_std:.4f}  batch_uniformity={batch_uniformity:.5f}"
        f"  n_harm_preds={len(harm_pred_errors)}  fatals={fatal}",
        flush=True,
    )

    return {
        "e1_count": e1_count,
        "e3_count": e3_count,
        "e1_per_e3": e1_per_e3,
        "e3_prediction_error_var": e3_pred_error_var,
        "harm_pred_std": harm_pred_std,
        "batch_uniformity": batch_uniformity,
        "n_harm_pred_samples": len(harm_pred_errors),
        "fatal_errors": fatal,
    }


def run(
    seed: int = 0,
    num_episodes: int = 300,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    lr: float = 1e-3,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:

    print(
        f"[V3-EXQ-066] MECH-089: Theta-Batching Reduction of E3 Prediction Error\n"
        f"  k={THETA_K}  episodes={num_episodes}  steps={steps_per_episode}\n"
        f"  alpha_world={alpha_world}  seed={seed}",
        flush=True,
    )

    # Create env params dict; instantiate separately for each condition so both
    # conditions start with an identical RNG state (matched design).
    _env_kwargs = dict(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    print(f"[V3-EXQ-066] Running condition A (batched: theta_buffer_size={THETA_K}, e3_steps={THETA_K})...", flush=True)
    batched_out = _run_condition(
        env=CausalGridWorldV2(**_env_kwargs),
        seed=seed,
        theta_buffer_size=THETA_K,
        e3_steps_per_tick=THETA_K,
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        alpha_world=alpha_world,
        lr=lr,
        self_dim=self_dim,
        world_dim=world_dim,
        label="batched",
    )

    print(f"[V3-EXQ-066] Running condition B (raw: theta_buffer_size=1, e3_steps=1)...", flush=True)
    raw_out = _run_condition(
        env=CausalGridWorldV2(**_env_kwargs),
        seed=seed,
        theta_buffer_size=1,
        e3_steps_per_tick=1,
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        alpha_world=alpha_world,
        lr=lr,
        self_dim=self_dim,
        world_dim=world_dim,
        label="raw",
    )

    e3_pred_var_batched = batched_out["e3_prediction_error_var"]
    e3_pred_var_raw = raw_out["e3_prediction_error_var"]

    # C1: batched error var < raw * 0.90 (at least 10% reduction)
    c1_pass = e3_pred_var_batched < e3_pred_var_raw * 0.90

    # C2: E1 ran at least k-1 times per E3 sample in the batched condition
    c2_pass = batched_out["e1_per_e3"] >= float(THETA_K - 1)

    # C3: E3 harm evaluator not collapsed in batched condition
    c3_pass = batched_out["harm_pred_std"] > 0.01

    # C4: E3 harm evaluator not collapsed in raw condition
    c4_pass = raw_out["harm_pred_std"] > 0.01

    # C5: no fatal errors in either condition
    c5_pass = (batched_out["fatal_errors"] == 0) and (raw_out["fatal_errors"] == 0)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: e3_pred_var_batched={e3_pred_var_batched:.5f} >= raw*0.90={e3_pred_var_raw*0.90:.5f}"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: e1_per_e3={batched_out['e1_per_e3']:.2f} < k-1={THETA_K-1}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: harm_pred_std_batched={batched_out['harm_pred_std']:.4f} <= 0.01"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: harm_pred_std_raw={raw_out['harm_pred_std']:.4f} <= 0.01"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: fatal_errors batched={batched_out['fatal_errors']} raw={raw_out['fatal_errors']}"
        )

    print(f"\nV3-EXQ-066 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        # Batched condition
        "e3_prediction_error_batched":     float(e3_pred_var_batched),
        "harm_pred_std_batched":            float(batched_out["harm_pred_std"]),
        "batch_uniformity":                 float(batched_out["batch_uniformity"]),
        "e1_per_e3_batched":                float(batched_out["e1_per_e3"]),
        "e3_count_batched":                 float(batched_out["e3_count"]),
        "e1_count_batched":                 float(batched_out["e1_count"]),
        "n_harm_pred_samples_batched":      float(batched_out["n_harm_pred_samples"]),
        "fatal_errors_batched":             float(batched_out["fatal_errors"]),
        # Raw condition
        "e3_prediction_error_raw":          float(e3_pred_var_raw),
        "harm_pred_std_raw":                float(raw_out["harm_pred_std"]),
        "batch_uniformity_raw":             float(raw_out["batch_uniformity"]),
        "e1_per_e3_raw":                    float(raw_out["e1_per_e3"]),
        "e3_count_raw":                     float(raw_out["e3_count"]),
        "e1_count_raw":                     float(raw_out["e1_count"]),
        "n_harm_pred_samples_raw":          float(raw_out["n_harm_pred_samples"]),
        "fatal_errors_raw":                 float(raw_out["fatal_errors"]),
        # Reduction ratio
        "e3_pred_error_reduction_ratio":    float(
            e3_pred_var_batched / max(1e-9, e3_pred_var_raw)
        ),
        # Design constants
        "theta_k":                          float(THETA_K),
        # Criteria
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

    summary_markdown = f"""# V3-EXQ-066 -- MECH-089: Theta-Batching Reduces E3 Prediction Error

**Status:** {status}
**Claims:** MECH-089
**Design:** k={THETA_K} (theta buffer size); batched (theta_buffer_size=4, e3_steps=4) vs raw (theta_buffer_size=1, e3_steps=1)
**alpha_world:** {alpha_world}
**Training:** {num_episodes} episodes x {steps_per_episode} steps
**Seed:** {seed}

## Motivation

MECH-089: E3 never sees raw E1 output -- it receives theta-cycle-averaged z_world
from ThetaBuffer. Averaging over k={THETA_K} E1 steps should filter within-cycle
noise, reducing E3's harm-prediction error variance vs per-step (raw) sampling.

This experiment isolates the averaging effect by holding all other factors constant
(same seed, same env, same training dynamics, same alpha_world).

## Results

| Metric | Batched (k={THETA_K}) | Raw (k=1) |
|--------|----------------------|-----------|
| e3_prediction_error_var | {e3_pred_var_batched:.5f} | {e3_pred_var_raw:.5f} |
| harm_pred_std | {batched_out['harm_pred_std']:.4f} | {raw_out['harm_pred_std']:.4f} |
| batch_uniformity (mean z_world intra-window var) | {batched_out['batch_uniformity']:.5f} | {raw_out['batch_uniformity']:.5f} |
| e1_per_e3 | {batched_out['e1_per_e3']:.2f} | {raw_out['e1_per_e3']:.2f} |
| e3_count | {batched_out['e3_count']} | {raw_out['e3_count']} |
| fatal_errors | {batched_out['fatal_errors']} | {raw_out['fatal_errors']} |

Reduction ratio (batched/raw): {metrics['e3_pred_error_reduction_ratio']:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: e3_pred_var_batched < raw*0.90 (>=10% reduction) | {"PASS" if c1_pass else "FAIL"} | {e3_pred_var_batched:.5f} vs {e3_pred_var_raw*0.90:.5f} |
| C2: e1_per_e3_batched >= k-1 = {THETA_K-1} (batching active) | {"PASS" if c2_pass else "FAIL"} | {batched_out['e1_per_e3']:.2f} |
| C3: harm_pred_std_batched > 0.01 (E3 not collapsed) | {"PASS" if c3_pass else "FAIL"} | {batched_out['harm_pred_std']:.4f} |
| C4: harm_pred_std_raw > 0.01 (E3 not collapsed) | {"PASS" if c4_pass else "FAIL"} | {raw_out['harm_pred_std']:.4f} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | batched={batched_out['fatal_errors']} raw={raw_out['fatal_errors']} |

Criteria met: {criteria_met}/5 -> **{status}**
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
        "fatal_error_count": float(batched_out["fatal_errors"] + raw_out["fatal_errors"]),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--episodes",    type=int,   default=300)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--harm-scale",  type=float, default=0.02)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
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
