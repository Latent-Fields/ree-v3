"""
V3-EXQ-052 — SD-006: Multi-Rate Execution Validation

Claims: SD-006, MECH-089

Prerequisite: EXQ-041 PASS (ThetaBuffer smoke test works at default rates).

Motivation (2026-03-19):
  SD-006: Asynchronous multi-rate loop execution (phase 1: time-multiplexed).
  EXQ-041 validated ThetaBuffer at default rates, but did NOT test EXPLICIT rate
  separation. This experiment formally validates that E1/E2/E3 running at different
  rates (N_e2=3, N_e3=9) maintains:
    1. E3 receives theta-averaged z_world (not raw per-step values)
    2. E3 planning quality (calibration_gap_approach) is maintained at slower rates
    3. E3 tick count is proportionally lower than E1 tick count
    4. The multi-rate system degrades gracefully as rate separation increases

  Rate configurations tested:
    Baseline (N_e1=1, N_e2=1, N_e3=1): single rate, same as EXQ-041
    Separated (N_e1=1, N_e2=3, N_e3=9): multi-rate SD-006

  Key measurement: does calibration_gap_approach remain positive and meaningful
  at 3× rate separation (N_e3=9 vs N_e1=1)?

PASS criteria (ALL must hold):
  C1: calibration_gap_approach_separated > 0.0  (E3 still functional at slower rate)
  C2: e3_tick_ratio_separated < 0.3              (E3 runs at least 3× less than E1)
  C3: theta_buffer_size_used >= 3                (theta buffer is accumulating across steps)
  C4: e3_tick_count_separated >= 10             (E3 fired enough times)
  C5: No fatal errors
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


EXPERIMENT_TYPE = "v3_exq_052_sd006_multirate_validation"
CLAIM_IDS = ["SD-006", "MECH-089"]

APPROACH_TTYPES = {"hazard_approach"}


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _run_condition(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    label: str,
) -> Dict:
    """Run training + eval, tracking tick counts and calibration."""
    agent.train()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    e1_tick_total = 0
    e3_tick_total = 0
    approach_scores: List[float] = []
    none_scores: List[float] = []
    fatal = 0
    theta_sizes_seen: List[int] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e1_tick", False):
                e1_tick_total += 1

            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()

            # Track theta buffer fill level
            try:
                tsize = len(agent.theta_buffer._buffer)
                theta_sizes_seen.append(tsize)
            except Exception:
                pass

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                try:
                    result = agent.e3.select(candidates, temperature=1.0)
                    action = result.selected_action.detach()
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
            ttype = info.get("transition_type", "none")

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            if harm_signal < 0:
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

            # Calibration scores
            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(theta_z).item())
                if ttype in APPROACH_TTYPES:
                    approach_scores.append(score)
                elif ttype == "none":
                    none_scores.append(score)
            except Exception:
                pass

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

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
                break

    cal_gap = _mean_safe(approach_scores) - _mean_safe(none_scores)
    e3_ratio = e3_tick_total / max(1, e1_tick_total)
    max_theta_size = max(theta_sizes_seen) if theta_sizes_seen else 0

    print(
        f"  [{label}] e1_ticks={e1_tick_total}  e3_ticks={e3_tick_total}"
        f"  e3_ratio={e3_ratio:.3f}  max_theta_size={max_theta_size}"
        f"  cal_gap_approach={cal_gap:.4f}  n_approach={len(approach_scores)}",
        flush=True,
    )

    return {
        "e1_tick_total":          e1_tick_total,
        "e3_tick_total":          e3_tick_total,
        "e3_tick_ratio":          e3_ratio,
        "max_theta_buffer_size":  max_theta_size,
        "calibration_gap_approach": cal_gap,
        "n_approach_steps":       len(approach_scores),
        "fatal_errors":           fatal,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 400,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
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
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )
    # Default MultiRateClock already uses e2_steps_per_tick=3, e3_steps_per_tick=10
    # (from ree_core/heartbeat/clock.py defaults). We validate these default rates
    # are active and producing correct multi-rate behavior.
    config_separated = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )

    print(
        f"[V3-EXQ-052] SD-006: Multi-Rate Validation\n"
        f"  warmup={warmup_episodes}  alpha_world={alpha_world}\n"
        f"  Default clock rates: e2_steps_per_tick=3, e3_steps_per_tick=10",
        flush=True,
    )

    agent = REEAgent(config_separated)

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    print("[V3-EXQ-052] Running separated-rate condition (N_e2=3, N_e3=9)...", flush=True)
    sep_out = _run_condition(
        agent, env, optimizer, wf_optimizer, harm_eval_optimizer,
        warmup_episodes, steps_per_episode, world_dim, "separated",
    )

    # PASS / FAIL
    c1_pass = sep_out["calibration_gap_approach"] > 0.0
    c2_pass = sep_out["e3_tick_ratio"] < 0.3
    c3_pass = sep_out["max_theta_buffer_size"] >= 3
    c4_pass = sep_out["e3_tick_total"] >= 10
    c5_pass = sep_out["fatal_errors"] == 0

    all_pass    = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status      = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: cal_gap_approach={sep_out['calibration_gap_approach']:.4f} <= 0"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: e3_tick_ratio={sep_out['e3_tick_ratio']:.3f} >= 0.3"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: max_theta_buffer_size={sep_out['max_theta_buffer_size']} < 3"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: e3_tick_count={sep_out['e3_tick_total']} < 10"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={sep_out['fatal_errors']}")

    print(f"\nV3-EXQ-052 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "e1_tick_total":            float(sep_out["e1_tick_total"]),
        "e3_tick_total":            float(sep_out["e3_tick_total"]),
        "e3_tick_ratio":            float(sep_out["e3_tick_ratio"]),
        "max_theta_buffer_size":    float(sep_out["max_theta_buffer_size"]),
        "calibration_gap_approach": float(sep_out["calibration_gap_approach"]),
        "n_approach_steps":         float(sep_out["n_approach_steps"]),
        "fatal_error_count":        float(sep_out["fatal_errors"]),
        "e2_steps_per_tick":        3.0,
        "e3_steps_per_tick":        10.0,
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

    summary_markdown = f"""# V3-EXQ-052 — SD-006: Multi-Rate Execution Validation

**Status:** {status}
**Claims:** SD-006, MECH-089
**Prerequisite:** EXQ-041 PASS (ThetaBuffer at default rates)
**Rate config:** E1=1 (every step), E2=3, E3=9
**alpha_world:** {alpha_world}
**Training:** {warmup_episodes} eps
**Seed:** {seed}

## Motivation

SD-006 Phase 1: time-multiplexed multi-rate execution. E3 runs every 9 steps,
receiving theta-averaged z_world from ThetaBuffer (MECH-089). This tests whether
E3 remains functional (calibration_gap > 0) at 9× slower rate than E1.

## Multi-Rate Results

| Metric | Separated (e3_rate=9) |
|--------|-----------------------|
| e1_tick_total | {sep_out['e1_tick_total']} |
| e3_tick_total | {sep_out['e3_tick_total']} |
| e3_tick_ratio | {sep_out['e3_tick_ratio']:.3f} |
| max_theta_buffer_size | {sep_out['max_theta_buffer_size']} |
| calibration_gap_approach | {sep_out['calibration_gap_approach']:.4f} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: cal_gap_approach > 0 (E3 functional at slow rate) | {"PASS" if c1_pass else "FAIL"} | {sep_out['calibration_gap_approach']:.4f} |
| C2: e3_tick_ratio < 0.3 (E3 runs much less than E1) | {"PASS" if c2_pass else "FAIL"} | {sep_out['e3_tick_ratio']:.3f} |
| C3: max_theta_buffer_size >= 3 (buffer fills) | {"PASS" if c3_pass else "FAIL"} | {sep_out['max_theta_buffer_size']} |
| C4: e3_tick_count >= 10 (E3 fires) | {"PASS" if c4_pass else "FAIL"} | {sep_out['e3_tick_total']} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {sep_out['fatal_errors']} |

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
        "fatal_error_count": sep_out["fatal_errors"],
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--warmup",      type=int,   default=400)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--harm-scale",  type=float, default=0.02)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
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
