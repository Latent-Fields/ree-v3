"""
V3-EXQ-005 — MECH-069 Error Signal Scale Separation

Claim: MECH-069 — Sensory prediction error (E1), motor-sensory error (E2), and
harm/goal error (E3) are incommensurable in scale and temporal structure. They
cannot be combined by a fixed scalar lambda without losing the independent signal
content of each channel.

Experimental logic:
  Run 200 episodes with RANDOM policy (maximum hazard exposure). At each step,
  record the scalar loss values for each channel:
    L_E1(t)  = E1 sensory prediction loss (MSE over latent)
    L_E2s(t) = E2 self-transition loss (MSE)
    L_E2w(t) = E2 world_forward loss (MSE)
    L_E3(t)  = E3 harm BCE loss (only on steps with harm buffer >= 4 examples)

  Compute across the full series:
    - Mean and std for each loss channel
    - Scale ratio: max_mean / min_mean across channels (incommensurable if > 10)
    - Pairwise Pearson correlations (incommensurable if |r| < 0.3)
    - Coefficient of variation (CV = std/mean) per channel

  ARC-021 / MECH-069 prediction:
    - Scale ratio >> 10 (E1 MSE >> E3 BCE in typical magnitude)
    - |r(E1, E3)| < 0.3 (prediction error and harm are not co-varying)
    - |r(E2, E3)| < 0.3 (motor error and harm are not co-varying)
    - High CV on E3 (harm signal is sparse and bursty vs smooth E1/E2)

PASS criteria:
  C1: scale_ratio > 5.0  (channels differ meaningfully in magnitude)
  C2: |corr(E1, E3)| < 0.3  (E1 and E3 are not co-varying)
  C3: |corr(E2w, E3)| < 0.3  (E2 world and E3 are not co-varying)
  C4: warmup harm events > 100  (enough E3 signal to compute correlation)
  C5: fatal_error_count == 0
  C6: n_E3_measurements >= 50  (enough E3 loss measurements)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple
import math
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_005_mech069_error_scale"
CLAIM_IDS = ["MECH-069", "ARC-021"]


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation between two lists of equal length."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return num / denom if denom > 1e-12 else 0.0


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def run(
    seed: int = 0,
    num_episodes: int = 200,
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
    agent = REEAgent(config)
    opt_e12 = optim.Adam(agent.parameters(), lr=lr)
    opt_e3  = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=1e-4)

    # Per-step loss series (aligned: E1, E2s, E2w recorded every step;
    # E3 recorded only when a harm-buffer batch is available — stored with step index)
    series_e1:  List[float] = []
    series_e2s: List[float] = []
    series_e2w: List[float] = []
    series_e3:  List[float] = []      # sparse — steps where E3 training fired
    series_e3_e1_aligned:  List[float] = []  # E1 at same steps as E3 fired
    series_e3_e2w_aligned: List[float] = []  # E2w at same steps as E3 fired

    world_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf:    List[torch.Tensor] = []
    no_harm_buf: List[torch.Tensor] = []
    total_harm = 0
    fatal_errors = 0

    agent.train()

    try:
        for ep in range(num_episodes):
            flat_obs, obs_dict = env.reset()
            agent.reset()
            z_world_prev = None
            action_prev  = None

            for step in range(steps_per_episode):
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()

                if z_world_prev is not None and action_prev is not None:
                    world_buf.append((z_world_prev.detach(), action_prev.detach(), latent.z_world.detach()))
                    if len(world_buf) > 500: world_buf = world_buf[-500:]

                action_idx = random.randint(0, env.action_dim - 1)
                action = _action_to_onehot(action_idx, env.action_dim, agent.device)
                agent._last_action = action
                z_world_prev = latent.z_world.detach()
                action_prev  = action.detach()

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                if harm_signal < 0:
                    total_harm += 1
                    harm_buf.append(latent.z_world.detach())
                    if len(harm_buf) > 500: harm_buf = harm_buf[-500:]
                else:
                    if step % 3 == 0:
                        no_harm_buf.append(latent.z_world.detach())
                        if len(no_harm_buf) > 500: no_harm_buf = no_harm_buf[-500:]

                # E1 + E2 losses (record before backward)
                e1_loss  = agent.compute_prediction_loss()
                e2s_loss = agent.compute_e2_loss()

                e2w_loss = e1_loss.new_zeros(())
                if len(world_buf) >= 4:
                    n = min(16, len(world_buf))
                    idxs = torch.randperm(len(world_buf))[:n].tolist()
                    zw_t, acts, zw_t1 = zip(*[world_buf[i] for i in idxs])
                    e2w_loss = F.mse_loss(
                        agent.e2.world_forward(torch.cat(zw_t), torch.cat(acts)),
                        torch.cat(zw_t1)
                    )

                e1_val  = float(e1_loss.item())
                e2s_val = float(e2s_loss.item())
                e2w_val = float(e2w_loss.item()) if len(world_buf) >= 4 else float("nan")

                series_e1.append(e1_val)
                series_e2s.append(e2s_val)
                if not math.isnan(e2w_val):
                    series_e2w.append(e2w_val)

                # E1 + E2 backward
                e12_total = e1_loss + e2s_loss + e2w_loss
                if e12_total.requires_grad:
                    opt_e12.zero_grad()
                    e12_total.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    opt_e12.step()

                # E3 harm backward — record loss when it fires
                n_h, n_nh = len(harm_buf), len(no_harm_buf)
                if n_h >= 4 and n_nh >= 4:
                    k = min(16, n_h, n_nh)
                    zw_h  = torch.cat([harm_buf[i]    for i in torch.randperm(n_h)[:k].tolist()])
                    zw_nh = torch.cat([no_harm_buf[i] for i in torch.randperm(n_nh)[:k].tolist()])
                    labels = torch.cat([torch.ones(k, 1, device=agent.device),
                                        torch.zeros(k, 1, device=agent.device)])
                    pred = agent.e3.harm_eval(torch.cat([zw_h, zw_nh]))
                    if not torch.isnan(pred).any():
                        e3_loss = F.binary_cross_entropy(pred.clamp(1e-6, 1-1e-6), labels)
                        e3_val  = float(e3_loss.item())

                        # Record E3 loss and the aligned E1/E2w at this same step
                        series_e3.append(e3_val)
                        series_e3_e1_aligned.append(e1_val)
                        if not math.isnan(e2w_val):
                            series_e3_e2w_aligned.append(e2w_val)
                        else:
                            series_e3_e2w_aligned.append(0.0)

                        opt_e3.zero_grad()
                        e3_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                        opt_e3.step()

                if done:
                    break

            if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
                print(f"  ep {ep+1}/{num_episodes}  harm={total_harm}  "
                      f"E3_measurements={len(series_e3)}", flush=True)

    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL: {traceback.format_exc()}", flush=True)

    # ---------------------------------------------------------------------------
    # Compute statistics
    # ---------------------------------------------------------------------------
    def _safe_mean(xs): return sum(xs) / len(xs) if xs else 0.0
    def _safe_std(xs):
        if len(xs) < 2: return 0.0
        m = _safe_mean(xs)
        return math.sqrt(sum((x - m)**2 for x in xs) / len(xs))
    def _safe_cv(xs):
        m = _safe_mean(xs)
        return _safe_std(xs) / m if m > 1e-12 else 0.0

    mean_e1  = _safe_mean(series_e1)
    mean_e2s = _safe_mean(series_e2s)
    mean_e2w = _safe_mean(series_e2w)
    mean_e3  = _safe_mean(series_e3)

    std_e1  = _safe_std(series_e1)
    std_e2w = _safe_std(series_e2w)
    std_e3  = _safe_std(series_e3)

    cv_e1  = _safe_cv(series_e1)
    cv_e2w = _safe_cv(series_e2w)
    cv_e3  = _safe_cv(series_e3)

    means = [m for m in [mean_e1, mean_e2s, mean_e2w, mean_e3] if m > 1e-12]
    scale_ratio = max(means) / min(means) if len(means) >= 2 else 1.0

    # Correlation only where E3 fired
    n_e3 = len(series_e3)
    corr_e1_e3  = _pearson(series_e3_e1_aligned,  series_e3) if n_e3 >= 10 else 0.0
    corr_e2w_e3 = _pearson(series_e3_e2w_aligned, series_e3) if n_e3 >= 10 else 0.0

    # PASS criteria
    crit1_pass = scale_ratio > 5.0
    crit2_pass = abs(corr_e1_e3) < 0.3
    crit3_pass = abs(corr_e2w_e3) < 0.3
    crit4_pass = total_harm > 100
    crit5_pass = fatal_errors == 0
    crit6_pass = n_e3 >= 50

    all_pass = all([crit1_pass, crit2_pass, crit3_pass, crit4_pass, crit5_pass, crit6_pass])
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([crit1_pass, crit2_pass, crit3_pass, crit4_pass, crit5_pass, crit6_pass])

    failure_notes = []
    if not crit1_pass: failure_notes.append(f"C1 FAIL: scale_ratio {scale_ratio:.2f} <= 5.0")
    if not crit2_pass: failure_notes.append(f"C2 FAIL: |corr(E1, E3)| {abs(corr_e1_e3):.3f} >= 0.3")
    if not crit3_pass: failure_notes.append(f"C3 FAIL: |corr(E2w, E3)| {abs(corr_e2w_e3):.3f} >= 0.3")
    if not crit4_pass: failure_notes.append(f"C4 FAIL: warmup harm events {total_harm} <= 100")
    if not crit5_pass: failure_notes.append(f"C5 FAIL: fatal_errors={fatal_errors}")
    if not crit6_pass: failure_notes.append(f"C6 FAIL: n_E3_measurements {n_e3} < 50")

    print(f"\nMECH-069 / V3-EXQ-005 verdict: {status}  ({criteria_met}/6)", flush=True)
    print(f"  scale_ratio={scale_ratio:.2f}  corr(E1,E3)={corr_e1_e3:.3f}  corr(E2w,E3)={corr_e2w_e3:.3f}", flush=True)
    print(f"  means: E1={mean_e1:.4f}  E2s={mean_e2s:.4f}  E2w={mean_e2w:.4f}  E3={mean_e3:.4f}", flush=True)
    print(f"  CVs:   E1={cv_e1:.2f}  E2w={cv_e2w:.2f}  E3={cv_e3:.2f}  n_E3={n_e3}", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-005 -- MECH-069 Error Signal Scale Separation

**Status:** {status}
**Episodes:** {num_episodes} x {steps_per_episode} steps, RANDOM policy
**Hazards:** 8 (num_hazards)
**Seed:** {seed}

## MECH-069 Prediction

E1 (sensory prediction), E2 (motor-sensory), and E3 (harm/goal) error signals
are incommensurable in scale and temporal structure. They cannot be combined by
a fixed scalar lambda without corrupting the independent signal content of each.

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: scale_ratio > 5.0 (max/min mean loss) | {"PASS" if crit1_pass else "FAIL"} | {scale_ratio:.2f} |
| C2: abs(corr(E1, E3)) < 0.3 | {"PASS" if crit2_pass else "FAIL"} | {abs(corr_e1_e3):.3f} |
| C3: abs(corr(E2w, E3)) < 0.3 | {"PASS" if crit3_pass else "FAIL"} | {abs(corr_e2w_e3):.3f} |
| C4: Harm events > 100 | {"PASS" if crit4_pass else "FAIL"} | {total_harm} |
| C5: No fatal errors | {"PASS" if crit5_pass else "FAIL"} | {fatal_errors} |
| C6: n_E3_measurements >= 50 | {"PASS" if crit6_pass else "FAIL"} | {n_e3} |

## Loss Channel Statistics

| Channel | Mean | Std | CV (std/mean) |
|---|---|---|---|
| E1 (sensory prediction) | {mean_e1:.5f} | {std_e1:.5f} | {cv_e1:.2f} |
| E2w (world_forward) | {mean_e2w:.5f} | {std_e2w:.5f} | {cv_e2w:.2f} |
| E3 (harm BCE) | {mean_e3:.5f} | {std_e3:.5f} | {cv_e3:.2f} |

**Scale ratio (max/min mean):** {scale_ratio:.2f}

## Pairwise Correlations (at steps where E3 fired)

| Pair | Pearson r | Incommensurable? |
|---|---|---|
| corr(E1, E3) | {corr_e1_e3:.3f} | {"YES" if abs(corr_e1_e3) < 0.3 else "NO"} |
| corr(E2w, E3) | {corr_e2w_e3:.3f} | {"YES" if abs(corr_e2w_e3) < 0.3 else "NO"} |

Total harm events: {total_harm} | E3 measurements: {n_e3}

Criteria met: {criteria_met}/6 -> **{status}**
{failure_section}
"""

    metrics = {
        "fatal_error_count": float(fatal_errors),
        "total_harm_events": float(total_harm),
        "mean_e1_loss": mean_e1,
        "mean_e2s_loss": mean_e2s,
        "mean_e2w_loss": mean_e2w,
        "mean_e3_loss": mean_e3,
        "cv_e1": cv_e1,
        "cv_e2w": cv_e2w,
        "cv_e3": cv_e3,
        "scale_ratio": scale_ratio,
        "corr_e1_e3": corr_e1_e3,
        "corr_e2w_e3": corr_e2w_e3,
        "n_e3_measurements": float(n_e3),
        "crit1_pass": 1.0 if crit1_pass else 0.0,
        "crit2_pass": 1.0 if crit2_pass else 0.0,
        "crit3_pass": 1.0 if crit3_pass else 0.0,
        "crit4_pass": 1.0 if crit4_pass else 0.0,
        "crit5_pass": 1.0 if crit5_pass else 0.0,
        "crit6_pass": 1.0 if crit6_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

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
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps",    type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        num_episodes=args.episodes,
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
