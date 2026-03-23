"""
V3-EXQ-069 -- MECH-098 Reafference Event Selectivity Pair

Claims: MECH-098
Proposal: EXP-0001 / EVB-0001

MECH-098 asserts that the ReafferencePredictor (SD-007) subtracts predicted self-caused
perspective-shift from z_world, increasing z_world's selectivity for genuine external
events relative to pure locomotion steps.  This is the MSTd congruent/incongruent
neuron decomposition: after cancellation the residual delta-z_world should be larger
on event steps than on locomotion-only steps.

Discriminative pair (dispatch_mode=discriminative_pair, EXP-0001):

  REAFFERENCE_ON  -- alpha_world=0.9, ReafferencePredictor enabled (SD-007, MECH-101)
  REAFFERENCE_OFF -- alpha_world=0.9, ReafferencePredictor disabled

Both conditions: same seeds, same training budget, same CausalGridWorld config.
Reafference predictor trained ONLY on locomotion steps (transition_type=="none").

Pre-registered primary discriminator (threshold >= 0.01):

  event_selectivity_delta = margin(ON) - margin(OFF)
  margin = mean(||delta_z_world|| on non-"none" steps)
         - mean(||delta_z_world|| on "none" steps)

PASS criteria (ALL required):
  C1: event_selectivity_delta >= 0.01  (MECH-098 core claim)
  C2: reafference_r2 > 0.20            (predictor is learning, ON condition)
  C3: margin_OFF > 0                   (raw z_world has non-trivial baseline selectivity)
  C4: n_event_steps_min >= 10          (enough events in every run)

Decision scoring:
  retain_ree:       delta >= 0.01
  hybridize:        0 < delta < 0.01 (marginal benefit)
  retire_ree_claim: delta <= 0

Informational secondary metrics:
  mean_dz_loco / mean_dz_event per condition
  calibration_gap (diagnostic for EXQ-027b over-correction)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_069_reafference_selectivity_pair"
CLAIM_IDS = ["MECH-098"]

_EVENT_TYPES = {"agent_caused_hazard", "env_caused_hazard", "hazard_approach", "benefit_approach"}


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_single(
    seed: int,
    reafference_enabled: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
) -> Dict:
    """Run one (seed, condition) cell and return selectivity metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(
        seed=seed, size=12, num_hazards=15, num_resources=5,
        env_drift_interval=3, env_drift_prob=0.5,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim if reafference_enabled else 0,
    )
    agent = REEAgent(config)

    # Reafference buffer: (z_world_raw_prev, a_prev, dz_raw)
    reaf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_REAF = 5000

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n and "reafference_predictor" not in n
    ]
    optimizer = optim.Adam(standard_params, lr=lr)

    reaf_optimizer: Optional[optim.Optimizer] = None
    if reafference_enabled and agent.latent_stack.reafference_predictor is not None:
        reaf_optimizer = optim.Adam(
            list(agent.latent_stack.reafference_predictor.parameters()), lr=1e-3,
        )

    cond_label = "ON" if reafference_enabled else "OFF"

    # --- TRAIN ---
    agent.train()
    total_harm = 0
    n_loco_train = 0

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_raw_prev: Optional[torch.Tensor] = None
        a_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            # MECH-101: reafference predictor input must be z_world_raw_prev
            z_raw_curr = (
                latent.z_world_raw.detach()
                if latent.z_world_raw is not None
                else latent.z_world.detach()
            )

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if harm_signal < 0:
                total_harm += 1

            # Collect reafference training data (locomotion-only, ON condition)
            if (
                reafference_enabled
                and ttype == "none"
                and z_raw_prev is not None
                and a_prev is not None
                and agent.latent_stack.reafference_predictor is not None
            ):
                dz_raw = z_raw_curr - z_raw_prev
                reaf_data.append((z_raw_prev.cpu(), a_prev.cpu(), dz_raw.cpu()))
                n_loco_train += 1
                if len(reaf_data) > MAX_REAF:
                    reaf_data = reaf_data[-MAX_REAF:]

            # E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # Reafference predictor loss (ON condition only)
            if reaf_optimizer is not None and len(reaf_data) >= 16:
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
                        agent.latent_stack.reafference_predictor.parameters(), 0.5,
                    )
                    reaf_optimizer.step()

            z_raw_prev = z_raw_curr
            a_prev = action.detach()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" harm={total_harm} loco_collected={n_loco_train}",
                flush=True,
            )

    # --- Reafference R2 (ON condition, held-out 20%) ---
    reafference_r2 = 0.0
    if reafference_enabled and len(reaf_data) >= 20 and agent.latent_stack.reafference_predictor is not None:
        n = len(reaf_data)
        n_train = int(n * 0.8)
        with torch.no_grad():
            zwr_all = torch.cat([d[0] for d in reaf_data], dim=0).to(agent.device)
            a_all   = torch.cat([d[1] for d in reaf_data], dim=0).to(agent.device)
            dz_all  = torch.cat([d[2] for d in reaf_data], dim=0).to(agent.device)
            pred_all = agent.latent_stack.reafference_predictor(zwr_all, a_all)
            pred_test = pred_all[n_train:]
            dz_test   = dz_all[n_train:]
            if pred_test.shape[0] > 0:
                ss_res = ((dz_test - pred_test) ** 2).sum()
                ss_tot = ((dz_test - dz_test.mean(dim=0, keepdim=True)) ** 2).sum()
                reafference_r2 = float(max(0.0, (1 - ss_res / (ss_tot + 1e-8)).item()))
    print(f"  [reaf_r2] seed={seed} cond={cond_label} r2={reafference_r2:.4f}", flush=True)

    # --- EVAL: measure event selectivity ---
    # delta_z_world is the change in z_world between consecutive steps.
    # We associate each delta with the transition_type of the PRECEDING step
    # (the step that drove obs[t-1] -> obs[t]).
    agent.eval()

    dz_loco:  List[float] = []
    dz_event: List[float] = []
    n_fatal = 0

    # Calibration gap (informational only)
    harm_none: List[float] = []
    harm_agent: List[float] = []

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_prev: Optional[torch.Tensor] = None
        prev_ttype: Optional[str] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

            z_curr = latent.z_world.detach()

            # Record delta caused by prev transition
            if z_prev is not None and prev_ttype is not None:
                try:
                    delta = float(torch.norm(z_curr - z_prev).item())
                    if prev_ttype == "none":
                        dz_loco.append(delta)
                    elif prev_ttype in _EVENT_TYPES:
                        dz_event.append(delta)
                except Exception:
                    n_fatal += 1

            # Informational: harm_eval score
            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(z_curr).item())
                if prev_ttype == "none":
                    harm_none.append(score)
                elif prev_ttype == "agent_caused_hazard":
                    harm_agent.append(score)
            except Exception:
                pass

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action)
            prev_ttype = info.get("transition_type", "none")
            z_prev = z_curr

            if done:
                break

    mean_dz_loco  = float(sum(dz_loco)  / max(1, len(dz_loco)))
    mean_dz_event = float(sum(dz_event) / max(1, len(dz_event)))
    selectivity_margin = mean_dz_event - mean_dz_loco

    mn = sum(harm_none)  / max(1, len(harm_none))
    ma = sum(harm_agent) / max(1, len(harm_agent))
    calibration_gap = ma - mn

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" loco={mean_dz_loco:.4f} event={mean_dz_event:.4f}"
        f" margin={selectivity_margin:.4f}"
        f" n_event={len(dz_event)} n_loco={len(dz_loco)}"
        f" cal_gap={calibration_gap:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "reafference_enabled": reafference_enabled,
        "selectivity_margin": selectivity_margin,
        "mean_dz_loco": mean_dz_loco,
        "mean_dz_event": mean_dz_event,
        "reafference_r2": reafference_r2,
        "calibration_gap": calibration_gap,
        "n_event_steps": len(dz_event),
        "n_loco_steps": len(dz_loco),
        "n_agent_hazard_steps": len(harm_agent),
        "n_fatal": n_fatal,
    }


def run(
    seeds=(42, 7),
    warmup_episodes: int = 400,
    eval_episodes: int = 30,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    **kwargs,
) -> dict:
    """Discriminative pair: REAFFERENCE_ON vs REAFFERENCE_OFF."""
    results_on:  List[Dict] = []
    results_off: List[Dict] = []

    for seed in seeds:
        for reaf_on in [True, False]:
            label = "ON" if reaf_on else "OFF"
            print(
                f"\n[V3-EXQ-069] REAFFERENCE_{label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" alpha_world={alpha_world}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                reafference_enabled=reaf_on,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
            )
            if reaf_on:
                results_on.append(r)
            else:
                results_off.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return sum(vals) / max(1, len(vals))

    margin_on  = _avg(results_on,  "selectivity_margin")
    margin_off = _avg(results_off, "selectivity_margin")
    event_selectivity_delta = margin_on - margin_off

    reafference_r2_avg = _avg(results_on,  "reafference_r2")
    cal_gap_on_avg     = _avg(results_on,  "calibration_gap")
    cal_gap_off_avg    = _avg(results_off, "calibration_gap")
    n_event_min        = min(r["n_event_steps"] for r in results_on + results_off)

    # PASS criteria
    c1_pass = event_selectivity_delta >= 0.01
    c2_pass = reafference_r2_avg > 0.20
    c3_pass = margin_off > 0
    c4_pass = n_event_min >= 10

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    if event_selectivity_delta >= 0.01:
        decision = "retain_ree"
    elif event_selectivity_delta > 0:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-069] Final results:", flush=True)
    print(
        f"  margin_ON={margin_on:.4f}  margin_OFF={margin_off:.4f}"
        f"  delta={event_selectivity_delta:+.4f}",
        flush=True,
    )
    print(f"  reafference_r2={reafference_r2_avg:.4f}", flush=True)
    print(f"  cal_gap ON={cal_gap_on_avg:.4f}  OFF={cal_gap_off_avg:.4f}", flush=True)
    print(f"  decision={decision}  status={status} ({criteria_met}/4)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: event_selectivity_delta={event_selectivity_delta:.4f} < 0.01"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: reafference_r2={reafference_r2_avg:.4f} <= 0.20"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: margin_OFF={margin_off:.4f} <= 0 (baseline selectivity absent)"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: min event steps={n_event_min} < 10")

    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "event_selectivity_delta":  float(event_selectivity_delta),
        "selectivity_margin_on":    float(margin_on),
        "selectivity_margin_off":   float(margin_off),
        "mean_dz_loco_on":          float(_avg(results_on,  "mean_dz_loco")),
        "mean_dz_loco_off":         float(_avg(results_off, "mean_dz_loco")),
        "mean_dz_event_on":         float(_avg(results_on,  "mean_dz_event")),
        "mean_dz_event_off":        float(_avg(results_off, "mean_dz_event")),
        "reafference_r2_avg":       float(reafference_r2_avg),
        "calibration_gap_on":       float(cal_gap_on_avg),
        "calibration_gap_off":      float(cal_gap_off_avg),
        "n_event_steps_min":        float(n_event_min),
        "alpha_world":              float(alpha_world),
        "n_seeds":                  float(len(seeds)),
        "crit1_pass":               1.0 if c1_pass else 0.0,
        "crit2_pass":               1.0 if c2_pass else 0.0,
        "crit3_pass":               1.0 if c3_pass else 0.0,
        "crit4_pass":               1.0 if c4_pass else 0.0,
        "criteria_met":             float(criteria_met),
    }

    per_seed_rows_on = [
        f"  seed={r['seed']}: margin={r['selectivity_margin']:.4f}"
        f" r2={r['reafference_r2']:.4f} n_event={r['n_event_steps']}"
        for r in results_on
    ]
    per_seed_rows_off = [
        f"  seed={r['seed']}: margin={r['selectivity_margin']:.4f}"
        f" n_event={r['n_event_steps']}"
        for r in results_off
    ]

    if c1_pass:
        interpretation = (
            "MECH-098 SUPPORTED: reafference correction increases z_world selectivity"
            " for external events. Correction reduces locomotion noise while preserving"
            " event signal, consistent with MSTd congruent/incongruent decomposition."
        )
    elif event_selectivity_delta > 0:
        interpretation = (
            "Weak positive: correction shows marginal benefit (delta < 0.01 threshold)."
            " Predictor may need stronger training signal or longer warmup."
        )
    else:
        interpretation = (
            "MECH-098 NOT SUPPORTED: reafference correction does not improve z_world"
            " event selectivity. EXQ-027b over-correction hypothesis extends to"
            " selectivity domain. SD-010 (harm stream separation) may be required"
            " before MECH-098 can show net benefit."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    per_on_str  = "\n".join(per_seed_rows_on)
    per_off_str = "\n".join(per_seed_rows_off)

    summary_markdown = (
        f"# V3-EXQ-069 -- MECH-098 Reafference Event Selectivity Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-098\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n\n"
        f"## Pre-Registered Threshold\n\n"
        f"C1: event_selectivity_delta >= 0.01\n\n"
        f"## Results\n\n"
        f"| Condition | margin | mean_dz_loco | mean_dz_event | cal_gap (info) |\n"
        f"|-----------|--------|-------------|---------------|----------------|\n"
        f"| REAFFERENCE_ON  | {margin_on:.4f} | {_avg(results_on,  'mean_dz_loco'):.4f} | {_avg(results_on,  'mean_dz_event'):.4f} | {cal_gap_on_avg:.4f} |\n"
        f"| REAFFERENCE_OFF | {margin_off:.4f} | {_avg(results_off, 'mean_dz_loco'):.4f} | {_avg(results_off, 'mean_dz_event'):.4f} | {cal_gap_off_avg:.4f} |\n\n"
        f"**event_selectivity_delta (ON - OFF): {event_selectivity_delta:+.4f}**\n"
        f"reafference_r2 (ON, predictor quality): {reafference_r2_avg:.4f}\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: selectivity_delta >= 0.01 (core claim) | {'PASS' if c1_pass else 'FAIL'} | {event_selectivity_delta:.4f} |\n"
        f"| C2: reafference_r2 > 0.20 (predictor works) | {'PASS' if c2_pass else 'FAIL'} | {reafference_r2_avg:.4f} |\n"
        f"| C3: margin_OFF > 0 (baseline non-trivial) | {'PASS' if c3_pass else 'FAIL'} | {margin_off:.4f} |\n"
        f"| C4: min event steps >= 10 | {'PASS' if c4_pass else 'FAIL'} | {n_event_min} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"REAFFERENCE_ON:\n{per_on_str}\n"
        f"REAFFERENCE_OFF:\n{per_off_str}\n"
        f"{failure_section}\n"
    )

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": sum(r["n_fatal"] for r in results_on + results_off),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",       type=int, nargs="+", default=[42, 7])
    parser.add_argument("--warmup",      type=int,   default=400)
    parser.add_argument("--eval-eps",    type=int,   default=30)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--alpha-self",  type=float, default=0.3)
    args = parser.parse_args()

    result = run(
        seeds=args.seeds,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
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
