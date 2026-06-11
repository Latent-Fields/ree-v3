"""
V3-EXQ-671 -- MECH-025b: Precision-Responsibility Attribution Linkage

Claims: MECH-025b

Motivation (2026-06-11):
  MECH-025b: "High-precision action mode carries responsibility attribution:
  the precision level at which an action was committed determines the degree
  of ethical accountability assigned to its consequences."

  Decomposed from MECH-025 (2026-04-02). MECH-025 established that doing mode
  produces distinct internal signatures (EXQ-050 PASS). MECH-025b extends this:
  not just binary committed/uncommitted, but DEGREE of precision should modulate
  responsibility weight.

  Experimental design:
    - Train agent with functional E2 (world_forward) + E3 (harm_eval + precision)
    - During eval, track precision at each committed step (E3.current_precision)
    - Measure residue accumulation per step (ResidueField changes)
    - Compare residue-per-harm in high-precision vs low-precision regimes

  Key metrics:
    precision_residue_correlation: Pearson r(precision, residue_accumulated)
    high_precision_residue_ratio: mean residue/harm when precision > median
                                   vs mean residue/harm when precision <= median

  PASS criteria (ALL must hold):
    C1: precision_residue_correlation > 0.15  (positive correlation exists)
    C2: high_precision_residue_ratio > 1.1    (high-precision steps accumulate
                                                proportionally more residue)
    C3: committed_step_count >= 20             (sufficient samples)
    C4: world_forward_r2 > 0.05                (E2 attribution functional)
    C5: harm_pred_std > 0.01                   (E3 not collapsed)
    C6: No fatal errors
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
from experiment_protocol import emit_outcome


EXPERIMENT_TYPE = "v3_exq_671_mech025b_precision_responsibility"
CLAIM_IDS = ["MECH-025b"]


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denom_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    if denom_x < 1e-9 or denom_y < 1e-9:
        return 0.0
    return numerator / ((denom_x * denom_y) ** 0.5)


def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """Standard full-pipeline training to get functional E3 + E2.world_forward."""
    agent.train()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_harm = 0
    e3_tick_total = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action
            else:
                action = agent._last_action
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            if harm_signal < 0:
                total_harm += 1
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat(
                    [
                        torch.ones(k_p, 1, device=agent.device),
                        torch.zeros(k_n, 1, device=agent.device),
                    ],
                    dim=0,
                )
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}"
                f"  e3_ticks={e3_tick_total}",
                flush=True,
            )

    return {"total_harm": total_harm, "wf_buf": wf_buf, "e3_tick_total": e3_tick_total}


def _compute_world_forward_r2(agent: REEAgent, wf_buf: List, n_test: int = 200) -> float:
    if len(wf_buf) < n_test:
        return 0.0
    idxs = list(range(len(wf_buf) - n_test, len(wf_buf)))
    with torch.no_grad():
        zw = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
        a = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
        zw1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
        pred = agent.e2.world_forward(zw, a)
        ss_res = ((zw1 - pred) ** 2).sum()
        ss_tot = ((zw1 - zw1.mean(dim=0, keepdim=True)) ** 2).sum()
    return float((1 - ss_res / (ss_tot + 1e-8)).item())


def _eval_precision_responsibility(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """
    Probe precision-responsibility linkage by tracking:
      - precision at each committed step (E3.current_precision)
      - residue accumulated per step (ResidueField.total_residue delta)
      - harm magnitude per step

    Tests MECH-025b: high-precision actions should accumulate proportionally
    more residue (responsibility weight) than low-precision actions.
    """
    agent.eval()
    precision_samples: List[float] = []
    residue_delta_samples: List[float] = []
    harm_magnitude_samples: List[float] = []
    all_harm_preds: List[float] = []
    fatal = 0
    committed_step_count = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        residue_prev = float(agent.residue_field.total_residue.item())

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            try:
                if ticks.get("e3_tick", False) and candidates:
                    with torch.no_grad():
                        result = agent.e3.select(candidates, temperature=1.0)
                        action = result.selected_action.detach()
                        agent._last_action = action
                else:
                    action = agent._last_action
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action

                is_committed = agent._committed_candidates is not None

                # Track precision at committed steps
                if is_committed:
                    with torch.no_grad():
                        current_precision = agent.e3.current_precision
                        h_pred = float(agent.e3.harm_eval(latent.z_world).item())
                        all_harm_preds.append(h_pred)

                # Execute action
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                # Track residue accumulation
                residue_curr = float(agent.residue_field.total_residue.item())
                residue_delta = residue_curr - residue_prev
                residue_prev = residue_curr

                # Collect samples for committed steps with harm
                if is_committed and abs(harm_signal) > 1e-6:
                    committed_step_count += 1
                    precision_samples.append(current_precision)
                    residue_delta_samples.append(residue_delta)
                    harm_magnitude_samples.append(abs(harm_signal))

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                residue_prev = float(agent.residue_field.total_residue.item())

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

    # Compute correlation and ratio
    precision_residue_correlation = _pearson_correlation(precision_samples, residue_delta_samples)

    # High vs low precision comparison
    if len(precision_samples) >= 4:
        median_precision = sorted(precision_samples)[len(precision_samples) // 2]
        high_prec_residue = []
        low_prec_residue = []
        for i in range(len(precision_samples)):
            if precision_samples[i] > median_precision:
                high_prec_residue.append(residue_delta_samples[i] / max(harm_magnitude_samples[i], 1e-6))
            else:
                low_prec_residue.append(residue_delta_samples[i] / max(harm_magnitude_samples[i], 1e-6))

        mean_high = _mean_safe(high_prec_residue)
        mean_low = _mean_safe(low_prec_residue)
        high_precision_residue_ratio = mean_high / max(mean_low, 1e-6)
    else:
        high_precision_residue_ratio = 0.0

    harm_pred_std = (
        float(torch.tensor(all_harm_preds).std().item()) if len(all_harm_preds) > 1 else 0.0
    )

    print(
        f"  precision_residue_correlation={precision_residue_correlation:.4f}"
        f"  high_precision_residue_ratio={high_precision_residue_ratio:.4f}"
        f"  committed_steps={committed_step_count}",
        flush=True,
    )

    return {
        "precision_residue_correlation": precision_residue_correlation,
        "high_precision_residue_ratio": high_precision_residue_ratio,
        "committed_step_count": committed_step_count,
        "harm_pred_std": harm_pred_std,
        "fatal_errors": fatal,
        "n_samples": len(precision_samples),
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 500,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    lr: float = 1e-3,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
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

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()),
        lr=1e-4,
    )

    print(
        f"[V3-EXQ-671] MECH-025b: Precision-Responsibility Attribution\n"
        f"  warmup={warmup_episodes}  eval={eval_episodes}  alpha_world={alpha_world}",
        flush=True,
    )

    train_out = _train(
        agent,
        env,
        optimizer,
        wf_optimizer,
        harm_eval_optimizer,
        warmup_episodes,
        steps_per_episode,
        world_dim,
    )
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_buf"])
    print(f"  world_forward_r2: {world_forward_r2:.4f}", flush=True)

    print(f"\n[V3-EXQ-671] Eval -- probing precision-responsibility linkage...", flush=True)
    eval_out = _eval_precision_responsibility(
        agent, env, eval_episodes, steps_per_episode, world_dim
    )

    # PASS / FAIL
    c1_pass = eval_out["precision_residue_correlation"] > 0.15
    c2_pass = eval_out["high_precision_residue_ratio"] > 1.1
    c3_pass = eval_out["committed_step_count"] >= 20
    c4_pass = world_forward_r2 > 0.05
    c5_pass = eval_out["harm_pred_std"] > 0.01
    c6_pass = eval_out["fatal_errors"] == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass and c6_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: precision_residue_correlation="
            f"{eval_out['precision_residue_correlation']:.4f} <= 0.15"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: high_precision_residue_ratio="
            f"{eval_out['high_precision_residue_ratio']:.4f} <= 1.1"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: committed_step_count={eval_out['committed_step_count']} < 20"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: world_forward_r2={world_forward_r2:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: harm_pred_std={eval_out['harm_pred_std']:.4f} <= 0.01"
        )
    if not c6_pass:
        failure_notes.append(f"C6 FAIL: fatal_errors={eval_out['fatal_errors']}")

    print(f"\nV3-EXQ-671 verdict: {status}  ({criteria_met}/6)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "precision_residue_correlation": float(eval_out["precision_residue_correlation"]),
        "high_precision_residue_ratio": float(eval_out["high_precision_residue_ratio"]),
        "committed_step_count": float(eval_out["committed_step_count"]),
        "harm_pred_std": float(eval_out["harm_pred_std"]),
        "world_forward_r2": float(world_forward_r2),
        "e3_tick_total": float(train_out["e3_tick_total"]),
        "total_harm_train": float(train_out["total_harm"]),
        "fatal_error_count": float(eval_out["fatal_errors"]),
        "n_samples": float(eval_out["n_samples"]),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "crit6_pass": 1.0 if c6_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-671 -- MECH-025b: Precision-Responsibility Attribution

**Status:** {status}
**Claim:** MECH-025b -- high-precision action mode carries responsibility attribution
**Prerequisite:** MECH-025 (doing mode produces internal signature)
**alpha_world:** {alpha_world}
**Warmup:** {warmup_episodes} eps  |  Eval: {eval_episodes} eps
**Seed:** {seed}

## Motivation

MECH-025b tests the philosophical bridge: does precision level modulate
responsibility weight? Actions committed at higher precision should accumulate
proportionally more residue (ethical accountability) than low-precision actions,
because high-precision implies the agent had finer discrimination capacity.

## Results

| Metric | Value |
|--------|-------|
| Precision-Residue Correlation | {eval_out['precision_residue_correlation']:.4f} |
| High/Low Precision Residue Ratio | {eval_out['high_precision_residue_ratio']:.4f} |
| Committed Steps Sampled | {eval_out['committed_step_count']} |
| World Forward R2 | {world_forward_r2:.4f} |
| Harm Pred Std | {eval_out['harm_pred_std']:.4f} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: precision_residue_correlation > 0.15 | {"PASS" if c1_pass else "FAIL"} | {eval_out['precision_residue_correlation']:.4f} |
| C2: high_precision_residue_ratio > 1.1 | {"PASS" if c2_pass else "FAIL"} | {eval_out['high_precision_residue_ratio']:.4f} |
| C3: committed_step_count >= 20 | {"PASS" if c3_pass else "FAIL"} | {eval_out['committed_step_count']} |
| C4: world_forward_r2 > 0.05 | {"PASS" if c4_pass else "FAIL"} | {world_forward_r2:.4f} |
| C5: harm_pred_std > 0.01 | {"PASS" if c5_pass else "FAIL"} | {eval_out['harm_pred_std']:.4f} |
| C6: No fatal errors | {"PASS" if c6_pass else "FAIL"} | {eval_out['fatal_errors']} |

Criteria met: {criteria_met}/6 -> **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if criteria_met >= 4 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": eval_out["fatal_errors"],
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--harm-scale", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true", help="Quick validation run")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Quick validation mode", flush=True)
        args.warmup = 5
        args.eval_eps = 2
        args.steps = 50

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
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

    if not args.dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly"
            / "evidence"
            / "experiments"
            / EXPERIMENT_TYPE
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        print(f"\nResult written to: {out_path}", flush=True)

    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)

    if not args.dry_run:
        _outcome_raw = str(result.get("status", "FAIL")).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=out_path,
        )
